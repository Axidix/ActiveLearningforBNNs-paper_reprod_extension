
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data import load_mnist, get_data_loaders
from src.models import PaperCNN
from src.novel_extension_task.matrixnormal_bayes_last_layer import MatrixNormalBayesianLastLayer


def to_one_hot(y, num_classes=10):
    return F.one_hot(y, num_classes=num_classes).float()



def eval_rmse_cached(Phi_test, Y_test, bayes: MatrixNormalBayesianLastLayer, params, head: str):
    with torch.no_grad():
        if head == "analytic":
            mean, _ = bayes.predictive(Phi_test, params, return_cov=False)
        else:
            mean, _, _ = bayes.mfvi_predictive_stats(Phi_test, params)
        diff2 = (mean - Y_test) ** 2
        return (diff2.sum().item() / diff2.numel()) ** 0.5

def eval_gaussian_nll_cached(Phi_test, Y_test, bayes: MatrixNormalBayesianLastLayer, params, head: str):
    # Efficient Gaussian NLL using analytic structure (analytic) or cached eig (MFVI)
    with torch.no_grad():
        D = Y_test.shape[1]
        if head == "analytic":
            mean, _ = bayes.predictive(Phi_test, params, return_cov=False)
            # Use analytic eigenbasis: P, s(x)
            P = params["Vinv_evecs"]  # (D,D)
            s = bayes._analytic_predictive_s_diag(Phi_test, params)  # (N,D)
            sigma2 = bayes.sigma2
            # For each test point, eigenvalues are sigma2 + s_d(x)
            nlls = []
            for i in range(mean.shape[0]):
                diff = (Y_test[i] - mean[i]).unsqueeze(0)  # (1,D)
                evals = sigma2 + s[i]  # (D,)
                evals = evals.clamp(min=1e-8)
                # Project diff into P basis
                diff_proj = diff @ P  # (1,D)
                quad = (diff_proj * diff_proj / evals).sum()
                logdet = torch.log(evals).sum()
                nll = 0.5 * (logdet + quad + D * np.log(2 * np.pi))
                nlls.append(nll.item())
            return float(np.mean(nlls))
        else:
            mean, c, V = bayes.mfvi_predictive_stats(Phi_test, params)
            # Eigendecompose V once
            evals_V, P = torch.linalg.eigh(V)  # (D,), (D,D)
            evals_V = evals_V.clamp(min=1e-8)
            sigma2 = bayes.sigma2
            nlls = []
            for i in range(mean.shape[0]):
                diff = (Y_test[i] - mean[i]).unsqueeze(0)  # (1,D)
                evals = sigma2 + c[i] * evals_V  # (D,)
                evals = evals.clamp(min=1e-8)
                diff_proj = diff @ P  # (1,D)
                quad = (diff_proj * diff_proj / evals).sum()
                logdet = torch.log(evals).sum()
                nll = 0.5 * (logdet + quad + D * np.log(2 * np.pi))
                nlls.append(nll.item())
            return float(np.mean(nlls))

def eval_accuracy_cached(Phi_test, Y_test, bayes: MatrixNormalBayesianLastLayer, params, head: str):
    # Predict class = argmax(mean), compare to true class
    with torch.no_grad():
        if head == "analytic":
            mean, _ = bayes.predictive(Phi_test, params, return_cov=False)
        else:
            mean, _, _ = bayes.mfvi_predictive_stats(Phi_test, params)
        pred = mean.argmax(dim=1)
        true = Y_test.argmax(dim=1)
        return (pred == true).float().mean().item()


def run_novel_extension_active_learning(
    heads=("analytic", "mfvi"),
    acq_strategies=None,
    num_acq_steps=50,
    acq_size=10,
    num_train_samples=20,
    num_val_samples=100,
    train_batch_size=8,
    pool_batch_size=256,
    test_batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    plot_root="plots/novel_extension_task",
    sigma2=1.0,
    s2=1.0,
    a=1.0,
    b_grid=(0.0, -0.05, -0.09),
    mfvi_iters=8,
    num_repeats=1,
    seed=42,
    num_pretrain_epochs = 10
):
    os.makedirs(plot_root, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    orig_trainset, testset, train_indices, val_indices, pool_indices = load_mnist(num_train_samples, num_val_samples)
    _, _, _, test_loader = get_data_loaders(
        orig_trainset, testset, train_indices, val_indices, pool_indices,
        train_batch_size, train_batch_size, pool_batch_size, test_batch_size
    )

    # Pretrain CNN feature extractor once
    model = PaperCNN(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader, val_loader, _, _ = get_data_loaders(
        orig_trainset, testset, train_indices, val_indices, pool_indices,
        train_batch_size, train_batch_size, pool_batch_size, test_batch_size
    )

    model.train()
    for i in range(num_pretrain_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"Pretrain loss (epoch {i}):", loss.item())

    # Freeze backbone
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Cache features
    print("Precomputing features...")
    all_loader = torch.utils.data.DataLoader(orig_trainset, batch_size=pool_batch_size, shuffle=False)

    Phi_all, Y_all = [], []
    for x, y in all_loader:
        x = x.to(device)
        Phi_all.append(model.forward_features(x))
        Y_all.append(to_one_hot(y.to(device), num_classes=10))
    Phi_all = torch.cat(Phi_all, dim=0)  # (N,K)
    Y_all = torch.cat(Y_all, dim=0)      # (N,D)

    Phi_test, Y_test = [], []
    for x, y in test_loader:
        x = x.to(device)
        Phi_test.append(model.forward_features(x))
        Y_test.append(to_one_hot(y.to(device), num_classes=10))
    Phi_test = torch.cat(Phi_test, dim=0)
    Y_test = torch.cat(Y_test, dim=0)


    eval_every = 5  # Evaluate NLL/acc every k rounds

    # Define acquisition strategies per head
    analytic_acq = ("trace_total", "logdet_total", "random", "trace_epistemic_norm", "logdet_total_norm", "logdet_epistemic_norm")
    mfvi_acq = ("trace_total", "logdet_total", "random", "trace_epistemic_norm", "logdet_total_norm", "logc_norm")

    for b in b_grid:
        exp_name = f"steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}_a{a}_b{b}_numtrainsamples{num_train_samples}_seed{seed}"
        plot_dir = os.path.join(plot_root, exp_name)
        os.makedirs(plot_dir, exist_ok=True)

        # Subfolders for metrics and plots
        metrics_root = os.path.join(plot_dir, "metrics")
        os.makedirs(metrics_root, exist_ok=True)
        metrics_dirs = {m: os.path.join(metrics_root, m) for m in ["rmse", "nll", "accuracy"]}
        for d in metrics_dirs.values():
            os.makedirs(d, exist_ok=True)
        plots_dirs = {m: os.path.join(plot_dir, m) for m in ["rmse", "nll", "accuracy"]}
        for d in plots_dirs.values():
            os.makedirs(d, exist_ok=True)

        results = {}
        acquired_labels = {}
        diag_info = {}

        for head in heads:
            if head == "analytic":
                acq_strategies_head = analytic_acq
            else:
                acq_strategies_head = mfvi_acq
            for acq in acq_strategies_head:
                key = f"{head}_{acq}"
                results[key] = {"num_labeled": [], "rmse": [], "nll": [], "acc": []}
                acquired_labels[key] = []
                diag_info[key] = {"offdiag_mass": []}

                for repeat in range(num_repeats):
                    labeled = list(train_indices)
                    pool = list(pool_indices)

                    bayes = MatrixNormalBayesianLastLayer(sigma2=sigma2, s2=s2, a=a, b=b)

                    # round 0
                    Phi_L = Phi_all[labeled]
                    Y_L = Y_all[labeled]

                    if head == "analytic":
                        params = bayes.fit_analytic(Phi_L, Y_L)
                    else:
                        params = bayes.fit_mfvi(Phi_L, Y_L, num_iters=mfvi_iters, verbose=True)

                    rmse = eval_rmse_cached(Phi_test, Y_test, bayes, params, head=head)
                    results[key]["num_labeled"].append(len(labeled))
                    results[key]["rmse"].append(rmse)
                    # NLL/acc only every eval_every rounds (always at round 0)
                    nll = acc = None
                    if 0 % eval_every == 0:
                        nll = eval_gaussian_nll_cached(Phi_test, Y_test, bayes, params, head=head)
                        acc = eval_accuracy_cached(Phi_test, Y_test, bayes, params, head=head)
                    results[key]["nll"].append(nll)
                    results[key]["acc"].append(acc)

                    if head == "mfvi":
                        diag_info[key]["offdiag_mass"].append(bayes.offdiag_mass(params["V"]).item())

                    print(f"[b={b}] {key} | Repeat {repeat+1} | Round 0 | Labeled: {len(labeled)} | RMSE: {rmse:.4f}")

                    for t in range(num_acq_steps):
                        Phi_U = Phi_all[pool]

                        if acq == "random":
                            scores = torch.rand(len(pool))
                        else:
                            if head == "analytic":
                                scores = bayes.acquisition_score(Phi_U, params, mode=acq)
                            else:
                                scores = bayes.acquisition_score_mfvi(Phi_U, params, mode=acq)

                        top_idx = torch.topk(scores, min(acq_size, len(pool))).indices.tolist()
                        selected = [pool[i] for i in top_idx]

                        acquired_labels[key].extend([orig_trainset[idx][1] for idx in selected])

                        labeled.extend(selected)
                        selected_set = set(selected)
                        pool = [idx for idx in pool if idx not in selected_set]

                        Phi_L = Phi_all[labeled]
                        Y_L = Y_all[labeled]

                        if head == "analytic":
                            params = bayes.fit_analytic(Phi_L, Y_L)
                        else:
                            params = bayes.fit_mfvi(Phi_L, Y_L, num_iters=mfvi_iters)

                        rmse = eval_rmse_cached(Phi_test, Y_test, bayes, params, head=head)
                        results[key]["num_labeled"].append(len(labeled))
                        results[key]["rmse"].append(rmse)
                        # NLL/acc only every eval_every rounds
                        nll = acc = None
                        if (t + 1) % eval_every == 0 or (t + 1) == num_acq_steps:
                            nll = eval_gaussian_nll_cached(Phi_test, Y_test, bayes, params, head=head)
                            acc = eval_accuracy_cached(Phi_test, Y_test, bayes, params, head=head)
                        results[key]["nll"].append(nll)
                        results[key]["acc"].append(acc)

                        if head == "mfvi":
                            diag_info[key]["offdiag_mass"].append(bayes.offdiag_mass(params["V"]).item())

                        if (t + 1) % 5 == 0 or (t + 1) == num_acq_steps:
                            print(f"[b={b}] {key} | Repeat {repeat+1} | Round {t+1} | Labeled: {len(labeled)} | RMSE: {rmse:.4f}")

        # --- plots
        # --- plots for each metric
        for metric, ylabel in zip(["rmse", "nll", "acc"], ["Test RMSE", "Test Gaussian NLL", "Test Accuracy"]):
            for head in heads:
                if head == "analytic":
                    acq_strategies_head = analytic_acq
                else:
                    acq_strategies_head = mfvi_acq
                plt.figure(figsize=(8, 6))
                for acq in acq_strategies_head:
                    key = f"{head}_{acq}"
                    if key not in results:
                        continue
                    xs = results[key]["num_labeled"]
                    ys = results[key][metric]
                    # Only plot non-None values for nll/acc
                    if metric == "rmse":
                        plt.plot(xs, ys, label=acq)
                    else:
                        xs_plot = [x for x, y in zip(xs, ys) if y is not None]
                        ys_plot = [y for y in ys if y is not None]
                        plt.plot(xs_plot, ys_plot, label=acq)
                plt.xlabel("Number of labeled samples")
                plt.ylabel(ylabel)
                plt.title(f"{head} inference | b={b} | acquisition comparison")
                plt.legend()
                plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dirs[metric], f"{metric}_{head}_b{b}.png"))
                plt.close()

            # Analytic vs MFVI (pick a couple strategies if available)
            for acq in [x for x in analytic_acq if x != "random"]:
                if f"analytic_{acq}" in results and f"mfvi_{acq}" in results:
                    plt.figure(figsize=(8, 6))
                    xs_a = results[f"analytic_{acq}"]["num_labeled"]
                    ys_a = results[f"analytic_{acq}"][metric]
                    xs_m = results[f"mfvi_{acq}"]["num_labeled"]
                    ys_m = results[f"mfvi_{acq}"][metric]
                    if metric == "rmse":
                        plt.plot(xs_a, ys_a, label=f"analytic_{acq}")
                        plt.plot(xs_m, ys_m, label=f"mfvi_{acq}")
                    else:
                        xs_a_plot = [x for x, y in zip(xs_a, ys_a) if y is not None]
                        ys_a_plot = [y for y in ys_a if y is not None]
                        xs_m_plot = [x for x, y in zip(xs_m, ys_m) if y is not None]
                        ys_m_plot = [y for y in ys_m if y is not None]
                        plt.plot(xs_a_plot, ys_a_plot, label=f"analytic_{acq}")
                        plt.plot(xs_m_plot, ys_m_plot, label=f"mfvi_{acq}")
                    plt.xlabel("Number of labeled samples")
                    plt.ylabel(ylabel)
                    plt.title(f"Analytic vs MFVI | {acq} | b={b}")
                    plt.legend()
                    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dirs[metric], f"analytic_vs_mfvi_{acq}_b{b}.png"))
                    plt.close()

        # Offdiag mass curve for MFVI (use first available mfvi key)
        mfvi_keys = [k for k in diag_info if k.startswith("mfvi") and len(diag_info[k]["offdiag_mass"]) > 0]
        if mfvi_keys:
            key = mfvi_keys[0]
            plt.figure(figsize=(8, 5))
            xs = results[key]["num_labeled"]
            ys = diag_info[key]["offdiag_mass"]
            plt.plot(xs, ys)
            plt.xlabel("Number of labeled samples")
            plt.ylabel("Off-diagonal mass in V (MFVI)")
            plt.title(f"MFVI output correlations | b={b} | {key}")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"mfvi_offdiag_mass_b{b}_{key}.png"))
            plt.close()

        # --- save metrics
        for key in results:
            # RMSE: always present
            arr = np.column_stack([results[key]["num_labeled"], results[key]["rmse"]])
            np.savetxt(os.path.join(metrics_dirs["rmse"], f"rmse_{key}.txt"), arr, header="num_labeled\trmse")
            # NLL/acc: only save non-None values
            xs_nll = [x for x, y in zip(results[key]["num_labeled"], results[key]["nll"]) if y is not None]
            ys_nll = [y for y in results[key]["nll"] if y is not None]
            if xs_nll:
                arr_nll = np.column_stack([xs_nll, ys_nll])
                np.savetxt(os.path.join(metrics_dirs["nll"], f"nll_{key}.txt"), arr_nll, header="num_labeled\tnll")
            xs_acc = [x for x, y in zip(results[key]["num_labeled"], results[key]["acc"]) if y is not None]
            ys_acc = [y for y in results[key]["acc"] if y is not None]
            if xs_acc:
                arr_acc = np.column_stack([xs_acc, ys_acc])
                np.savetxt(os.path.join(metrics_dirs["accuracy"], f"acc_{key}.txt"), arr_acc, header="num_labeled\tacc")

        # --- histograms
        for key, labels in acquired_labels.items():
            hist, _ = np.histogram(labels, bins=np.arange(11) - 0.5)
            np.savetxt(os.path.join(plot_dir, f"acquired_hist_{key}.txt"), hist, fmt="%d", header="class_histogram_0-9")
            plt.figure(figsize=(6, 4))
            plt.bar(np.arange(10), hist)
            plt.xlabel("Class label")
            plt.ylabel("Count")
            plt.title(f"Acquired label histogram: {key} | b={b}")
            plt.xticks(np.arange(10))
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"acquired_hist_{key}_b{b}.png"))
            plt.close()

        # Save simple diag info
        for key in diag_info:
            if len(diag_info[key]["offdiag_mass"]) == 0:
                continue
            np.savetxt(os.path.join(plot_dir, f"offdiag_mass_{key}.txt"),
                       np.asarray(diag_info[key]["offdiag_mass"]), header="offdiag_mass")

        print(f"Saved results to: {plot_dir}")


if __name__ == "__main__":
    run_novel_extension_active_learning(
        heads=("analytic", "mfvi"),
        num_acq_steps=100,
        acq_size=10,
        sigma2=1.0,
        num_train_samples=20,
        s2=1.0,
        a=1.0,
        b_grid=(-0.06, 0.0),
        mfvi_iters=5,
        num_repeats=1,
        seed=42,
        num_pretrain_epochs=15
    )
