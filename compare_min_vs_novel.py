import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from min_extension_active_learning import run_min_extension_active_learning
from novel_extension_active_learning import run_novel_extension_active_learning

def _exp_name_min(num_acq_steps, acq_size, sigma2, s2, num_train_samples, seed):
    return f"steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}_ntrain{num_train_samples}_seed{seed}"

def _exp_name_novel(num_acq_steps, acq_size, sigma2, s2, a, b, num_train_samples, seed):
    return f"steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}_a{a}_b{b}_numtrainsamples{num_train_samples}_seed{seed}"

def _load_metric_file(path):
    if not os.path.exists(path):
        return None, None
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, 0], arr[:, 1]

def _metric_path(root_dir, metric, key):
    if metric == "rmse":
        return os.path.join(root_dir, "metrics", "rmse", f"rmse_{key}.txt")
    if metric == "nll":
        return os.path.join(root_dir, "metrics", "nll", f"nll_{key}.txt")
    if metric == "acc":
        return os.path.join(root_dir, "metrics", "accuracy", f"acc_{key}.txt")
    raise ValueError(metric)

def _plot_overlay(out_path, title, xlabel, ylabel, curves):
    plt.figure(figsize=(8, 6))
    for c in curves:
        if c["x"] is None or c["y"] is None or len(c["x"]) == 0:
            continue
        plt.plot(c["x"], c["y"], label=c["label"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compare_min_vs_novel(
    heads=("analytic", "mfvi"),
    num_acq_steps=100,
    acq_size=10,
    num_train_samples=20,
    sigma2=1.0,
    s2=1.0,
    seed=42,
    num_pretrain_epochs=15,
    min_root="plots/min_extension_task",
    novel_root="plots/novel_extension_task",
    out_root="plots/compare_extensions",
    a=1.0,
    b=-0.06,
    mfvi_iters=5,
    run_min=True,
    run_novel=True,
    n_repeat=1
):
    # Acquisition strategies as in the actual experiment scripts
    min_acq_analytic = ("trace_total", "trace_epistemic_norm", "random")
    min_acq_mfvi = ("trace_total", "trace_epistemic_norm", "random")
    novel_acq_analytic = ("trace_total", "logdet_total", "random", "trace_epistemic_norm", "logdet_total_norm", "logdet_epistemic_norm")
    novel_acq_mfvi = ("trace_total", "logdet_total", "random", "trace_epistemic_norm", "logdet_total_norm", "logc_norm")

    # Repeat experiments n_repeat times with different seeds, aggregate metrics
    min_dirs = []
    novel_dirs = []
    seeds = [seed + i for i in range(n_repeat)]
    if run_min:
        for rep_seed in seeds:
            for head in heads:
                acq_strategies = min_acq_analytic if head == "analytic" else min_acq_mfvi
                run_min_extension_active_learning(
                    heads=(head,),
                    acq_strategies=acq_strategies,
                    num_acq_steps=num_acq_steps,
                    acq_size=acq_size,
                    num_train_samples=num_train_samples,
                    sigma2=sigma2,
                    s2=s2,
                    num_repeats=1,
                    seed=rep_seed,
                )
    if run_novel:
        for rep_seed in seeds:
            for head in heads:
                acq_strategies = novel_acq_analytic if head == "analytic" else novel_acq_mfvi
                run_novel_extension_active_learning(
                    heads=(head,),
                    acq_strategies=acq_strategies,
                    num_acq_steps=num_acq_steps,
                    acq_size=acq_size,
                    num_train_samples=num_train_samples,
                    sigma2=sigma2,
                    s2=s2,
                    a=a,
                    b_grid=(b,),
                    mfvi_iters=mfvi_iters,
                    num_repeats=1,
                    seed=rep_seed,
                    num_pretrain_epochs=num_pretrain_epochs,
                )

    # Collect all experiment directories for each repeat
    for rep_seed in seeds:
        min_pattern = os.path.join(min_root, f"steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}*seed{rep_seed}*")
        min_cands = sorted(glob.glob(min_pattern))
        if len(min_cands) == 0:
            raise FileNotFoundError(f"Minimal results dir not found (pattern: {min_pattern})")
        min_dirs.append(min_cands[-1])

        novel_pattern = os.path.join(novel_root, f"steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}_a{a}_b{b}*seed{rep_seed}*")
        novel_cands = sorted(glob.glob(novel_pattern))
        if len(novel_cands) == 0:
            raise FileNotFoundError(f"Novel results dir not found (pattern: {novel_pattern})")
        novel_dirs.append(novel_cands[-1])

    compare_name = f"min_vs_novel__steps{num_acq_steps}_acq{acq_size}_sigma2{sigma2}_s2{s2}_a{a}_b{b}_seed{seed}_nrep{n_repeat}__mfviIters{mfvi_iters}"
    out_dir = os.path.join(out_root, compare_name)
    os.makedirs(out_dir, exist_ok=True)
    metric_info = {
        "rmse": ("Test RMSE", "lower is better"),
        "nll": ("Test Gaussian NLL", "lower is better"),
        "acc": ("Test Accuracy", "higher is better"),
    }
    for metric, (ylabel, _) in metric_info.items():
        metric_dir = os.path.join(out_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        for head in heads:
            min_acqs = min_acq_analytic if head == "analytic" else min_acq_mfvi
            novel_acqs = novel_acq_analytic if head == "analytic" else novel_acq_mfvi
            acqs = sorted(set(min_acqs).intersection(set(novel_acqs)))
            for acq in acqs:
                # Aggregate metrics for each repeat
                min_xs, min_ys = [], []
                nov_xs, nov_ys = [], []
                for min_dir, novel_dir in zip(min_dirs, novel_dirs):
                    key_min = f"{head}_{acq}"
                    key_novel = f"{head}_{acq}"
                    x_min, y_min = _load_metric_file(_metric_path(min_dir, metric, key_min))
                    x_nov, y_nov = _load_metric_file(_metric_path(novel_dir, metric, key_novel))
                    if x_min is not None and y_min is not None:
                        min_xs.append(x_min)
                        min_ys.append(y_min)
                    if x_nov is not None and y_nov is not None:
                        nov_xs.append(x_nov)
                        nov_ys.append(y_nov)
                # Average and std over repeats (assume x is the same for all repeats)
                def avg_std(xs, ys):
                    if len(xs) == 0:
                        return None, None, None
                    x = xs[0]
                    ymat = np.stack([y for y in ys if y is not None])
                    y_mean = ymat.mean(axis=0)
                    y_std = ymat.std(axis=0)
                    return x, y_mean, y_std
                x_min, y_min_mean, y_min_std = avg_std(min_xs, min_ys)
                x_nov, y_nov_mean, y_nov_std = avg_std(nov_xs, nov_ys)
                curves = []
                if x_min is not None:
                    curves.append({"label": f"minimal | {acq}", "x": x_min, "y": y_min_mean})
                    curves.append({"label": f"minimal | {acq} ± std", "x": x_min, "y": y_min_mean + y_min_std, "style": "dashed", "alpha": 0.3})
                    curves.append({"label": f"minimal | {acq} ∓ std", "x": x_min, "y": y_min_mean - y_min_std, "style": "dashed", "alpha": 0.3})
                if x_nov is not None:
                    curves.append({"label": f"novel (a={a}, b={b}) | {acq}", "x": x_nov, "y": y_nov_mean})
                    curves.append({"label": f"novel (a={a}, b={b}) | {acq} ± std", "x": x_nov, "y": y_nov_mean + y_nov_std, "style": "dashed", "alpha": 0.3})
                    curves.append({"label": f"novel (a={a}, b={b}) | {acq} ∓ std", "x": x_nov, "y": y_nov_mean - y_nov_std, "style": "dashed", "alpha": 0.3})
                # Custom plot_overlay to handle style/alpha
                def plot_overlay_adv(out_path, title, xlabel, ylabel, curves):
                    plt.figure(figsize=(8, 6))
                    for c in curves:
                        if c["x"] is None or c["y"] is None or len(c["x"]) == 0:
                            continue
                        style = c.get("style", "solid")
                        alpha = c.get("alpha", 1.0)
                        plt.plot(c["x"], c["y"], label=c["label"], linestyle=style, alpha=alpha)
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out_path)
                    plt.close()
                title = f"{metric.upper()} overlay | head={head} | acq={acq}"
                out_path = os.path.join(metric_dir, f"{metric}_overlay_{head}_{acq}.png")
                plot_overlay_adv(out_path, title, "Number of labeled samples", ylabel, curves)
            # Summary curves (mean only)
            summary_curves = []
            for acq in acqs:
                min_xs, min_ys = [], []
                nov_xs, nov_ys = [], []
                for min_dir, novel_dir in zip(min_dirs, novel_dirs):
                    key_min = f"{head}_{acq}"
                    key_novel = f"{head}_{acq}"
                    x_min, y_min = _load_metric_file(_metric_path(min_dir, metric, key_min))
                    x_nov, y_nov = _load_metric_file(_metric_path(novel_dir, metric, key_novel))
                    if x_min is not None and y_min is not None:
                        min_xs.append(x_min)
                        min_ys.append(y_min)
                    if x_nov is not None and y_nov is not None:
                        nov_xs.append(x_nov)
                        nov_ys.append(y_nov)
                x_min, y_min_mean, _ = avg_std(min_xs, min_ys)
                x_nov, y_nov_mean, _ = avg_std(nov_xs, nov_ys)
                if x_min is not None:
                    summary_curves.append({"label": f"min | {acq}", "x": x_min, "y": y_min_mean})
                if x_nov is not None:
                    summary_curves.append({"label": f"nov | {acq}", "x": x_nov, "y": y_nov_mean})
            title = f"{metric.upper()} summary | head={head} | min vs novel (a={a}, b={b})"
            out_path = os.path.join(metric_dir, f"{metric}_summary_{head}.png")
            _plot_overlay(out_path, title, "Number of labeled samples", ylabel, summary_curves)
    hist_dir = os.path.join(out_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    with open(os.path.join(hist_dir, "histogram_files.txt"), "w") as f:
        for p in sorted(glob.glob(os.path.join(min_dir, "acquired_hist_*.png"))):
            f.write(f"MIN\t{p}\n")
        for p in sorted(glob.glob(os.path.join(novel_dir, "acquired_hist_*_b*.png"))):
            f.write(f"NOV\t{p}\n")
    print("==== Comparison complete ====")
    print("Minimal dir:", min_dir)
    print("Novel dir:", novel_dir)
    print("Compare dir:", out_dir)

if __name__ == "__main__":
    compare_min_vs_novel(
        heads=("analytic", "mfvi"),
        num_acq_steps=100,
        acq_size=10,
        num_train_samples=20,
        sigma2=1.0,
        s2=1.0,
        seed=42,
        num_pretrain_epochs=15,
        a=1.0,
        b=-0.06,
        mfvi_iters=10,
        run_min=True,
        run_novel=True,
        n_repeat=5
    )
