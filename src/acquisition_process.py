import torch

def max_entropy(predictions):
    """Max entropy acquisition function.
    Predictions shape expected: (T, num_samples, num_classes) as torch tensor"""

    avg_probs = predictions.mean(dim=0) # Shape: (num_samples, num_classes)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1) # Shape: (num_samples,)

    return entropy


def variation_ratios(predictions):
    """Variation Ratios acquisition function.
    Predictions shape expected: (T, num_samples, num_classes) as torch tensor"""

    # Get predicted classes for each pred
    avg_probs = predictions.mean(dim=0)  # Shape: (num_samples, num_classes)
    
    return 1 - avg_probs.max(dim=1).values  # Shape: (num_samples,)


def BALD(predictions):
    """BALD acquisition function.
    Predictions shape expected: (T, num_samples, num_classes) as torch tensor"""

    if predictions.shape[0] == 1:
        # Deterministic model case: return zeros
        return torch.zeros(predictions.shape[1])

    entropy = max_entropy(predictions)  # (num_samples,)
    expected_entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=2).mean(dim=0)  # Shape: (num_samples,)

    return entropy - expected_entropy


def mean_std(predictions):
    """Mean Standard Deviation acquisition function.
    Predictions shape expected: (T, num_samples, num_classes) as torch tensor"""

    exp_of_squares = (predictions ** 2).mean(dim=0)  # Shape: (num_samples, num_classes)
    expectation = predictions.mean(dim=0)
    var = exp_of_squares - expectation ** 2
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)

    return std.mean(dim=1)  # (num_samples,)


def random_acquisition(predictions):
    """Random acquisition function.
    Predictions shape expected: (T, num_samples, num_classes) as torch tensor.
    We generate random scores for each sample to match generic acquisition function template."""

    num_samples = predictions.shape[1]
    return torch.rand(num_samples)


def get_acquisition_function(name):
    """Returns acquisition function based on name."""

    if name == "entropy":
        return max_entropy
    elif name == "variation_ratios":
        return variation_ratios
    elif name == "bald":
        return BALD
    elif name == "mean_std":
        return mean_std
    elif name == "random":
        return random_acquisition
    else:
        raise ValueError(f"Acquisition function '{name}' not recognized.")
    
    
def get_acquisition_points(acq_function, predictions, num_points=10):
    """Selects points based on acquisition function scores.
    predictions: torch tensor of shape (T, num_samples, num_classes)
    num_points: number of points to acquire
    Returns indices of selected points."""

    scores = acq_function(predictions)  # Shape: (num_samples,)
    # For deterministic bald, scores will all be the same so take random indices
    if scores.max() - scores.min() < 1e-4:
        perm = torch.randperm(scores.size(0))
        selected_indices = perm[:num_points]
    else:
        _, selected_indices = torch.topk(scores, num_points)
    return selected_indices



def acquire_and_add_points(
    model,
    T,
    pool_loader,
    acquisition_function,
    num_points,
    device,
    train_indices,
    pool_indices,
    bayesian=True,
):
    """Acquire new points from the pool set using the specified acquisition function and add them to the training set.
    Uses MC Dropout with T forward passes to get predictions."""

    # Map from pool-loader order back to original dataset indices (pool_loader.dataset is a Subset)
    pool_dataset_indices = pool_loader.dataset.indices
    pool_size = len(pool_dataset_indices)
    if pool_size == 0:
        return train_indices, pool_indices

    k = min(int(num_points), pool_size)

    # Random acquisition: no need to compute predictions at all
    if acquisition_function is random_acquisition:
        top_indices = torch.randperm(pool_size)[:k].tolist()

    else:
        # MC-dropout vs deterministic mode
        if bayesian:
            model.train()   # keep dropout on
        else:
            model.eval()    # dropout off

        preds_list = []
        with torch.no_grad():
            for data, _ in pool_loader:
                data = data.to(device)
                B = data.size(0)

                # Expand without materializing T copies (expand is a view)
                data_rep = data.unsqueeze(0).expand(T, *data.shape).reshape(T * B, *data.shape[1:])

                logits = model(data_rep)
                probs = torch.softmax(logits, dim=1).view(T, B, -1)  # (T, B, C)
                preds_list.append(probs.cpu())

        all_predictions = torch.cat(preds_list, dim=1)  # (T, N, C)
        print("Predictions computed.")

        # Your existing selection logic (keeps tie-handling)
        top_indices = get_acquisition_points(acquisition_function, all_predictions, num_points=k).tolist()

    # Convert selected pool positions to original dataset indices
    acquired_indices = [pool_dataset_indices[i] for i in top_indices]

    # Add acquired indices to training set
    train_indices.extend(acquired_indices)

    # Remove acquired indices from pool set (stable order, avoids set() scrambling)
    acquired_set = set(acquired_indices)
    pool_indices[:] = [i for i in pool_indices if i not in acquired_set]

    print("Acquired points added to training set.")
    return train_indices, pool_indices