import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist(num_train_samples=20, num_val_samples=100):
    """Load MNIST dataset and create data loaders for training, validation, and testing.
    Divides the training set into a small balanced training set, a validation set, and a pool set as
    described in the paper. Full test set is used for testing."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std for MNIST - classic normalization
    ])
    
    orig_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    targets = orig_trainset.targets.numpy()
    num_classes = 10    # MNIST classes

    # Select a random balanced initial training set (1/10 of num_train_samples per class)
    train_indices = []
    for c in range(num_classes):
        idx = (targets == c).nonzero()[0]
        idx = torch.from_numpy(idx)
        selected = idx[torch.randperm(len(idx))[:num_train_samples // num_classes]].tolist()
        train_indices.extend(selected)

    remaining_indices = list(set(range(len(orig_trainset))) - set(train_indices))
    remaining_indices = torch.tensor(remaining_indices)[torch.randperm(len(remaining_indices))].tolist()

    # Validation set (no mention of it being balanced)
    val_indices = remaining_indices[:num_val_samples]
    # Pool set: all remaining points
    pool_indices = remaining_indices[num_val_samples:]

    return orig_trainset, testset, train_indices, val_indices, pool_indices


def get_data_loaders(orig_trainset, testset, train_indices, val_indices, pool_indices, 
                     train_batch_size=8, val_batch_size=32, pool_batch_size=256, test_batch_size=256):
    """Create data loaders needed for training, validation, pool, and testing, using the correct indices."""

    train_subset = torch.utils.data.Subset(orig_trainset, train_indices)
    val_subset = torch.utils.data.Subset(orig_trainset, val_indices)
    pool_subset = torch.utils.data.Subset(orig_trainset, pool_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    pool_loader = torch.utils.data.DataLoader(pool_subset, batch_size=pool_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, pool_loader, test_loader











# Test
if __name__ == "__main__":
    orig_trainset, testset, train_indices, val_indices, pool_indices = load_mnist()
    print(f"Original training set size: {len(orig_trainset)}")
    print(f"Test set size: {len(testset)}")
    print(f"Initial training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Pool set size: {len(pool_indices)}")

    # Check class distrib
    train_labels = orig_trainset.targets[train_indices]
    for c in range(10):
        print(f"Class {c} in initial training set: {(train_labels == c).sum().item()} samples")    # Check class distribution in validation set
    val_labels = orig_trainset.targets[val_indices]
    for c in range(10):
        print(f"Class {c} in validation set: {(val_labels == c).sum().item()} samples")