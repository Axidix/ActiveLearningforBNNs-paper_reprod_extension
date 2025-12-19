import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from src.data import load_mnist, get_data_loaders
from src.models import PaperCNN
from src.min_extension_task.bayes_last_layer import BayesianLastLayer
from src.model_pipelines import eval_rmse
import random

def to_one_hot(y, num_classes=10):
    return F.one_hot(y, num_classes=num_classes).float()

def run_min_extension_active_learning(
    heads=("analytic", "mfvi"),
    acq_strategies=("trace_total", "trace_epistemic_norm", "random"),
    num_acq_steps=100,
    acq_size=10,
    num_train_samples=20,
    num_val_samples=100,
    train_batch_size=8,
    pool_batch_size=256,
    test_batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    plot_dir="plots/min_extension_task",
    sigma2=1.0,
    s2=1.0,
    acq_mode="trace",
    num_repeats=1,
    seed=42
):
    # Create a subfolder for this experiment's params
    exp_name = f"steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}_ntrain{num_train_samples}_seed{seed}"
    plot_dir = os.path.join(plot_dir, exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    orig_trainset, testset, train_indices, val_indices, pool_indices = load_mnist(num_train_samples, num_val_samples)
    _, _, _, test_loader = get_data_loaders(
        orig_trainset, testset, train_indices, val_indices, pool_indices,
        train_batch_size, train_batch_size, pool_batch_size, test_batch_size
    )
    # Pretrain CNN feature extractor
    model = PaperCNN(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader, _, _ = get_data_loaders(
        orig_trainset, testset, train_indices, val_indices, pool_indices,
        train_batch_size, train_batch_size, pool_batch_size, test_batch_size
    )
    num_pretrain_epochs = 10
    model.train()
    for epoch in range(num_pretrain_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Freeze feature extractor
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Precompute all features for orig_trainset and testset
    print("Precomputing features for all data...")
    all_loader = torch.utils.data.DataLoader(orig_trainset, batch_size=pool_batch_size, shuffle=False)
    Phi_all = []
    Y_all = []
    for x, y in all_loader:
        x = x.to(device)
        Phi_all.append(model.forward_features(x).to(device))
        Y_all.append(to_one_hot(y.to(device), num_classes=10))
    Phi_all = torch.cat(Phi_all, dim=0)  # (N, K)
    Y_all = torch.cat(Y_all, dim=0)      # (N, 10)
    test_features = []
    test_targets = []
    for x, y in test_loader:
        x = x.to(device)
        test_features.append(model.forward_features(x).to(device))
        test_targets.append(to_one_hot(y.to(device), num_classes=10))
    Phi_test = torch.cat(test_features, dim=0)
    Y_test = torch.cat(test_targets, dim=0)
    
    # Main experiment loop
    results = {}
    acquired_labels = {}  # For class histograms
    for head in heads:
        for acq in acq_strategies:
            key = f"{head}_{acq}"
            results[key] = {"num_labeled": [], "rmse": []}
            for repeat in range(num_repeats):
                labeled = list(train_indices)
                pool = list(pool_indices)
                bayes = BayesianLastLayer(sigma2=sigma2, s2=s2)
                # For class histogram
                acquired_labels[key] = []
                # Round 0 eval
                Phi_L = Phi_all[labeled]
                Y_L = Y_all[labeled]
                if head == "analytic":
                    params = bayes.fit_analytic(Phi_L, Y_L)
                elif head == "mfvi":
                    params = bayes.fit_mfvi(Phi_L, Y_L)
                rmse = eval_rmse_cached(Phi_test, Y_test, bayes, params, device)
                results[key]["num_labeled"].append(len(labeled))
                results[key]["rmse"].append(rmse)
                print(f"{key} | Repeat {repeat+1} | Round 0 | Labeled: {len(labeled)} | RMSE: {rmse:.4f}")
                for acq_round in range(num_acq_steps):
                    Phi_U = Phi_all[pool]
                    if acq == "random":
                        scores = torch.rand(len(pool))
                    else:
                        scores = bayes.acquisition_score(Phi_U, params, mode=acq)
                    top_idx = torch.topk(scores, acq_size).indices.tolist()
                    selected = [pool[i] for i in top_idx]
                    # Log acquired labels for histogram
                    acquired_labels[key].extend([orig_trainset[pool[i]][1] for i in top_idx])
                    labeled.extend(selected)
                    selected_set = set(selected)
                    pool = [idx for idx in pool if idx not in selected_set]
                    # Refit
                    Phi_L = Phi_all[labeled]
                    Y_L = Y_all[labeled]
                    if head == "analytic":
                        params = bayes.fit_analytic(Phi_L, Y_L)
                    elif head == "mfvi":
                        params = bayes.fit_mfvi(Phi_L, Y_L)
                    rmse = eval_rmse_cached(Phi_test, Y_test, bayes, params, device)
                    results[key]["num_labeled"].append(len(labeled))
                    results[key]["rmse"].append(rmse)
                    print(f"{key} | Repeat {repeat+1} | Round {acq_round+1} | Labeled: {len(labeled)} | RMSE: {rmse:.4f}")

    # Plot A: analytic_trace vs mfvi_trace
    plt.figure(figsize=(8, 6))
    for head in ["analytic", "mfvi"]:
        key = f"{head}_trace_total"
        if key in results:
            plt.plot(results[key]["num_labeled"], results[key]["rmse"], label=key)
    plt.xlabel("Number of labeled samples")
    plt.ylabel("Test RMSE")
    plt.title("Analytic vs MFVI (Trace Acquisition)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "analytic_vs_mfvi_trace.png")
    plt.savefig(plot_path)
    print(f"Plot A saved to {plot_path}")

    # Plot B: compare all acquisition methods for each inference
    for head in ["analytic", "mfvi"]:
        plt.figure(figsize=(8, 6))
        for acq in ["trace_total", "trace_epistemic_norm", "random"]:
            key = f"{head}_{acq}"
            if key in results:
                plt.plot(results[key]["num_labeled"], results[key]["rmse"], label=acq)
        plt.xlabel("Number of labeled samples")
        plt.ylabel("Test RMSE")
        plt.title(f"{head.capitalize()} Inference: Acquisition Comparison")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{head}_acquisition_comparison.png")
        plt.savefig(plot_path)
        print(f"Plot B saved to {plot_path}")

    # Save metrics
    for key in results:
        np.savetxt(os.path.join(plot_dir, f"rmse_{key}.txt"),
                   np.column_stack([results[key]["num_labeled"], results[key]["rmse"]]),
                   header="num_labeled\trmse")
    print(f"Metrics saved to {plot_dir}")

    # Save class histograms (text and plot)
    for key, labels in acquired_labels.items():
        hist, _ = np.histogram(labels, bins=np.arange(11)-0.5)
        np.savetxt(os.path.join(plot_dir, f"acquired_hist_{key}.txt"), hist, fmt='%d', header="class_histogram_0-9")
        # Plot histogram
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(10), hist, color='C0', alpha=0.8)
        plt.xlabel('Class label')
        plt.ylabel('Count')
        plt.title(f'Acquired label histogram: {key}')
        plt.xticks(np.arange(10))
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"acquired_hist_{key}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Histogram plot saved to {plot_path}")

# Efficient RMSE using cached features/targets
def eval_rmse_cached(Phi_test, Y_test, bayes, params, device):
    # Phi_test: (N, K), Y_test: (N, D)
    mean, _ = bayes.predictive(Phi_test, params)
    diff2 = (mean - Y_test) ** 2
    rmse = (diff2.sum().item() / diff2.numel()) ** 0.5
    return rmse

if __name__ == "__main__":
    std_grid = [0.1, 1, 10]
    train_samples_grid = [20, 100, 1000]
    for sigma2 in std_grid:
        for s2 in std_grid:
            for num_train_samples in train_samples_grid:
                print(f"Running: sigma2={sigma2}, s2={s2}, num_train_samples={num_train_samples}")
                run_min_extension_active_learning(
                    sigma2=sigma2,
                    s2=s2,
                    num_train_samples=num_train_samples
                )
