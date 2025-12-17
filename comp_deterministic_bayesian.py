
from src.experiment_acquisition import run_exp_deterministic, run_experiment
import matplotlib.pyplot as plt
import os
import numpy as np

PLOT_DIR = "plots"

if __name__ == "__main__":
    acq_functions = ["bald", "variation_ratios", "entropy", "mean_std", "random"]
    num_acq_steps = 100
    acq_size = 10
    num_repeats = 3
    num_epochs = 10

    os.makedirs(PLOT_DIR, exist_ok=True)

    histories_bayesian = {}
    histories_deterministic = {}

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

    # Plot
    for acq_fct in acq_functions:
        plt.figure(figsize=(8, 6))
        num_acquired_bayesian, mean_acc_bayesian = histories_bayesian[acq_fct]
        plt.plot(num_acquired_bayesian, mean_acc_bayesian, label=f"{acq_fct} (Bayesian)")
        num_acquired_det, mean_acc_det = histories_deterministic[acq_fct]
        plt.plot(num_acquired_det, mean_acc_det, label=f"{acq_fct} (Deterministic)")
        plt.xlabel("Number of acquired images")
        plt.ylabel("Test Accuracy (%)")
        plt.title(f"MNIST Test Accuracy vs. Number of Acquired Images ({acq_fct})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_file_path = os.path.join(PLOT_DIR, f"comparison_{acq_fct}_acq_fct.png")
        plt.savefig(plot_file_path)
        print(f"Plot saved to {plot_file_path}")

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
