
from src.experiment_acquisition import run_experiment
import matplotlib.pyplot as plt
import os
import numpy as np

PLOT_DIR = "plots"
PLOT_PATH = os.path.join(PLOT_DIR, "comparison_acq_fct.png")
METRIC_PATH = os.path.join(PLOT_DIR, "comparison_acq_fct_metrics.txt")

if __name__ == "__main__":
	acq_functions = ["bald", "variation_ratios", "entropy", "mean_std", "random"]
	num_acq_steps = 100
	acq_size = 10
	num_repeats = 3
	num_epochs = 10

	os.makedirs(PLOT_DIR, exist_ok=True)

	histories = {}
	histories_runs = {}
	for acq_fct in acq_functions:
		print(f"\nRunning acquisition function: {acq_fct}")
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
		histories[acq_fct] = (num_acquired, mean_acc)
		histories_runs[acq_fct] = all_histories

	# Plot
	plt.figure(figsize=(8, 6))
	for acq_label, (num_acquired, mean_acc) in histories.items():
		plt.plot(num_acquired, mean_acc, label=acq_label)
	plt.xlabel("Number of acquired images")
	plt.ylabel("Test Accuracy (%)")
	plt.title("MNIST Test Accuracy vs. Number of Acquired Images")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(PLOT_PATH)
	print(f"Plot saved to {PLOT_PATH}")

	# Save metrics to text file
	with open(METRIC_PATH, "w") as f:
		for acq_label, (num_acquired, mean_acc) in histories.items():
			f.write(f"Acquisition Function: {acq_label}\n")
			for n, acc in zip(num_acquired, mean_acc):
				f.write(f"{n}\t{acc:.4f}\n")
			f.write("\n")
    
	# Reproduce table 1: Number of acquired images to get to model error of % on MNIST.
	TABLE_PATH = os.path.join(PLOT_DIR, "comparison_acq_fct_table.txt")
	target_errors = [0.1, 0.05]
	with open(TABLE_PATH, "w") as table_file:
		table_file.write("Values are rounded up to the next integer (ceil) for each run.\n")
		for target_error in target_errors:
			target_acc = 100.0 * (1.0 - target_error)
			table_file.write(f"Target error: {target_error:.2f} (accuracy {target_acc:.2f}%)\n")
			print(f"Target error: {target_error:.2f} (accuracy {target_acc:.2f}%)")
			for acq_label in histories.keys():
				# For each run, get the number of images needed
				all_histories = np.array(histories_runs[acq_label])  # shape: (num_repeats, num_points, 2)
				n_needed = []
				for run in range(all_histories.shape[0]):
					num_acquired = all_histories[run, :, 0]
					mean_acc = all_histories[run, :, 1]
					test_err = 1 - mean_acc / 100.0
					indices = np.where(test_err <= target_error)[0]
					if len(indices) > 0:
						first_index = indices[0]
						if first_index == 0:
							num_images = num_acquired[first_index]
						else:
							# Linear interpolation between previous and current point
							x0, x1 = num_acquired[first_index - 1], num_acquired[first_index]
							y0, y1 = test_err[first_index - 1], test_err[first_index]
							if y1 == y0:
								num_images = x1
							else:
								num_images = x0 + (target_error - y0) * (x1 - x0) / (y1 - y0)
						n_needed.append(int(np.ceil(num_images)))
					else:
						n_needed.append(None)
				# Aggregate
				n_needed_valid = [n for n in n_needed if n is not None]
				if n_needed_valid:
					mean_n = np.mean(n_needed_valid)
					std_n = np.std(n_needed_valid, ddof=1) if len(n_needed_valid) > 1 else 0.0
					line = f"  {acq_label}: {int(np.ceil(mean_n))} ± {std_n:.1f} images"
				else:
					line = f"  {acq_label}: Did not reach target error {target_error:.2f}"
				print(line)
				table_file.write(line + "\n")
			table_file.write("\n")


