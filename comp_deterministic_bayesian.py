
from src.experiment_acquisition import run_exp_deterministic, run_experiment
import matplotlib.pyplot as plt
import os
import numpy as np

PLOT_DIR = "plots_acq_comparison"

if __name__ == "__main__":
    acq_functions = ["bald", "variation_ratios", "entropy"]
    num_acq_steps = 100
    acq_size = 10
    num_repeats = 3
    num_epochs = 10

    os.makedirs(PLOT_DIR, exist_ok=True)


    histories_bayesian = {}
    histories_deterministic = {}
    histories_runs = {}
    histories_runs_det = {}

    for acq_fct in acq_functions:
        print(f"\nRunning acquisition function for bayesian: {acq_fct}")
        all_histories = run_experiment(
            acq_fct,
            num_repeats=num_repeats,
            num_acq_steps=num_acq_steps,
            acq_size=acq_size,
            num_epochs=num_epochs,
            T=10
        )

        all_histories = np.array(all_histories)  # shape: (num_repeats, num_acq_steps, 2)
        mean_acc = all_histories[:, :, 1].mean(axis=0)
        num_train = all_histories[0, :, 0]
        num_acquired = num_train
        histories_bayesian[acq_fct] = (num_acquired, mean_acc)
        histories_runs[acq_fct] = all_histories

        print(f"\nRunning acquisition function for deterministic: {acq_fct}")
        all_histories_det = run_exp_deterministic(
            acq_fct,
            num_repeats=num_repeats,
            num_acq_steps=num_acq_steps,
            acq_size=acq_size,
            num_epochs=num_epochs
        )
        all_histories_det = np.array(all_histories_det)  # shape: (num_repeats, num_acq_steps, 2)
        mean_acc_det = all_histories_det[:, :, 1].mean(axis=0)
        num_train_det = all_histories_det[0, :, 0]
        num_acquired_det = num_train_det
        histories_deterministic[acq_fct] = (num_acquired_det, mean_acc_det)
        histories_runs_det[acq_fct] = all_histories_det


    # Plot individual plots with y-axis starting at 80%
    for acq_fct in acq_functions:
        plt.figure(figsize=(8, 6))
        # Bayesian
        all_histories_bayesian = histories_runs[acq_fct]  # shape: (num_repeats, num_points, 2)
        num_acquired_bayesian = all_histories_bayesian[0, :, 0]
        accs_bayesian = all_histories_bayesian[:, :, 1]
        mean_acc_bayesian = accs_bayesian.mean(axis=0)
        min_acc_bayesian = accs_bayesian.min(axis=0)
        max_acc_bayesian = accs_bayesian.max(axis=0)
        plt.plot(num_acquired_bayesian, mean_acc_bayesian, label=f"{acq_fct} (Bayesian)")
        plt.fill_between(num_acquired_bayesian, min_acc_bayesian, max_acc_bayesian, alpha=0.2, color='C0')
        # Deterministic
        all_histories_det = histories_runs_det[acq_fct]  # shape: (num_repeats, num_points, 2)
        num_acquired_det = all_histories_det[0, :, 0]
        accs_det = all_histories_det[:, :, 1]
        mean_acc_det = accs_det.mean(axis=0)
        min_acc_det = accs_det.min(axis=0)
        max_acc_det = accs_det.max(axis=0)
        plt.plot(num_acquired_det, mean_acc_det, label=f"{acq_fct} (Deterministic)")
        plt.fill_between(num_acquired_det, min_acc_det, max_acc_det, alpha=0.2, color='C1')
        plt.xlabel("Number of acquired images")
        plt.ylabel("Test Accuracy (%)")
        plt.title(f"MNIST Test Accuracy vs. Number of Acquired Images ({acq_fct})")
        plt.ylim(80, 100)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.minorticks_on()
        plt.tight_layout()
        plot_file_path = os.path.join(PLOT_DIR, f"comparison_{acq_fct}_acq_fct.png")
        plt.savefig(plot_file_path)
        print(f"Plot saved to {plot_file_path}")

    # Combined subplot: all three acquisition functions in one row
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    for idx, acq_fct in enumerate(acq_functions):
        ax = axes[idx]
        # Bayesian
        all_histories_bayesian = histories_runs[acq_fct]
        num_acquired_bayesian = all_histories_bayesian[0, :, 0]
        accs_bayesian = all_histories_bayesian[:, :, 1]
        mean_acc_bayesian = accs_bayesian.mean(axis=0)
        min_acc_bayesian = accs_bayesian.min(axis=0)
        max_acc_bayesian = accs_bayesian.max(axis=0)
        ax.plot(num_acquired_bayesian, mean_acc_bayesian, label=f"{acq_fct} (Bayesian)")
        ax.fill_between(num_acquired_bayesian, min_acc_bayesian, max_acc_bayesian, alpha=0.2, color='C0')
        # Deterministic
        all_histories_det = histories_runs_det[acq_fct]
        num_acquired_det = all_histories_det[0, :, 0]
        accs_det = all_histories_det[:, :, 1]
        mean_acc_det = accs_det.mean(axis=0)
        min_acc_det = accs_det.min(axis=0)
        max_acc_det = accs_det.max(axis=0)
        ax.plot(num_acquired_det, mean_acc_det, label=f"{acq_fct} (Deterministic)")
        ax.fill_between(num_acquired_det, min_acc_det, max_acc_det, alpha=0.2, color='C1')
        ax.set_xlabel("Number of acquired images")
        ax.set_title(f"{acq_fct}")
        ax.set_ylim(80, 100)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.minorticks_on()
        if idx == 0:
            ax.set_ylabel("Test Accuracy (%)")
        ax.legend()
    plt.suptitle("MNIST Test Accuracy vs. Number of Acquired Images (All Acquisition Functions)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_plot_path = os.path.join(PLOT_DIR, "comparison_all_acq_functions.png")
    plt.savefig(combined_plot_path)
    print(f"Combined plot saved to {combined_plot_path}")

    # Save metrics to text file
    metric_file_path = os.path.join(PLOT_DIR, "bay_vs_det_metrics.txt")
    with open(metric_file_path, "w") as f:
        for acq_fct in acq_functions:
            f.write(f"Acquisition Function: {acq_fct}\n")
            f.write("Bayesian:\n")
            num_acquired_bayesian, mean_acc_bayesian = histories_bayesian[acq_fct]
            for n, acc in zip(num_acquired_bayesian, mean_acc_bayesian):
                f.write(f"{n}\t{acc:.4f}\n")
            f.write("Deterministic:\n")
            num_acquired_det, mean_acc_det = histories_deterministic[acq_fct]
            for n, acc in zip(num_acquired_det, mean_acc_det):
                f.write(f"{n}\t{acc:.4f}\n")
            f.write("\n")
