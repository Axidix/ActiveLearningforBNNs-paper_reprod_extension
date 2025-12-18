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
    acq_strategies=("variance", "random"),
    num_acq_steps=20,
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
    # Main experiment loop: head ∈ {analytic, mfvi}, acq ∈ {variance, random}
    results = {}
    for head in heads:
        for acq in acq_strategies:
            key = f"{head}_{acq}"
            results[key] = {"num_labeled": [], "rmse": []}
            for repeat in range(num_repeats):
                labeled = list(train_indices)
                pool = list(pool_indices)
                bayes = BayesianLastLayer(sigma2=sigma2, s2=s2)
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
                        scores = bayes.acquisition_score(Phi_U, params, mode=acq_mode)
                    top_idx = torch.topk(scores, acq_size).indices.tolist()
                    selected = [pool[i] for i in top_idx]
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

    # Plot
    plt.figure(figsize=(8, 6))
    for key in results:
        plt.plot(results[key]["num_labeled"], results[key]["rmse"], label=key)
    plt.xlabel("Number of labeled samples")
    plt.ylabel("Test RMSE")
    plt.title("Active Learning with Bayesian Last Layer (Minimal Extension)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "min_extension_active_learning.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    # Save metrics
    for key in results:
        np.savetxt(os.path.join(plot_dir, f"rmse_{key}.txt"),
                   np.column_stack([results[key]["num_labeled"], results[key]["rmse"]]),
                   header="num_labeled\trmse")
    print(f"Metrics saved to {plot_dir}")

# Efficient RMSE using cached features/targets
def eval_rmse_cached(Phi_test, Y_test, bayes, params, device):
    # Phi_test: (N, K), Y_test: (N, D)
    mean, _ = bayes.predictive(Phi_test, params)
    diff2 = (mean - Y_test) ** 2
    rmse = (diff2.sum().item() / diff2.numel()) ** 0.5
    return rmse

if __name__ == "__main__":
    run_min_extension_active_learning()
