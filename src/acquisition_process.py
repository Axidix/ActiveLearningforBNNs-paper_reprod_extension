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

    entropy = max_entropy(predictions)  # Shape: (num_samples,)
    expected_entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=2).mean(dim=0)  # Shape: (num_samples,)

    return entropy - expected_entropy


def mean_std(predictions):
    """Mean Standard Deviation acquisition function.
    Predictions shape expected: (T, num_samples, num_classes) as torch tensor"""

    exp_of_squares = (predictions ** 2).mean(dim=0)  # Shape: (num_samples, num_classes)
    expectation = predictions.mean(dim=0)

    return torch.sqrt(exp_of_squares - expectation ** 2).mean(dim=1)  # Shape: (num_samples,)


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



def acquire_and_add_points(model, T, pool_loader, acquisition_function, num_points, device, train_indices, pool_indices, bayesian=True):
    """Acquire new points from the pool set using the specified acquisition function and add them to the training set.
    Uses MC Dropout with T forward passes to get predictions."""
    
    if bayesian: model.train()   # Keep dropout for MC Dropout
    else: model.eval()  # Deterministic model
    all_predictions = []

    with torch.no_grad():
        for data, _ in pool_loader:
            data = data.to(device)
            B = data.size(0)
            # Repeat batch T times for vectorized MC dropout
            data_repeat = data.repeat((T, 1, 1, 1))  # (T*B, ...)
            output = model(data_repeat)
            probs = torch.softmax(output, dim=1)
            # Reshape to (T, B, C)
            preds = probs.view(T, B, -1).cpu()
            all_predictions.append(preds)

    all_predictions = torch.cat(all_predictions, dim=1)  # Shape: (T, num_pool_samples, num_classes)

    # Get acquisition scores
    top_indices = get_acquisition_points(acquisition_function, all_predictions, num_points=num_points)
    top_indices = top_indices.tolist()

    # Map back to original dataset indices
    pool_dataset_indices = pool_loader.dataset.indices
    acquired_indices = [pool_dataset_indices[i] for i in top_indices]

    # Add acquired indices to training set
    train_indices.extend(acquired_indices)

    # Remove acquired indices from pool set
    pool_indices[:] = list(set(pool_indices) - set(acquired_indices))

    return train_indices, pool_indices