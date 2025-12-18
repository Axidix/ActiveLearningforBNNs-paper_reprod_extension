import torch

from src.data import load_mnist, get_data_loaders
from src.models import PaperCNN
from src.model_pipelines import train_model, evaluate_accuracy_mc_dropout, evaluate_accuracy_base
from src.acquisition_process import acquire_and_add_points, get_acquisition_function

def run_experiment_once(acq_function, num_acq_steps, acq_size, num_epochs, T):
    # Get MNIST data with initial split
    orig_trainset, testset, train_indices, val_indices, pool_indices = load_mnist(
        num_train_samples=20, num_val_samples=100
    )
    print("MNIST data loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_loader, val_loader, pool_loader, test_loader = get_data_loaders(
        orig_trainset, testset, train_indices, val_indices, pool_indices,
        train_batch_size=16, val_batch_size=16, pool_batch_size=128, test_batch_size=256
    )
    print("Initial data loaders created.")

    history = []  # To store (num_acquired, test_accuracy) after each acquisition step

    # Initial train/eval
    model = PaperCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    train_model(model, train_loader, val_loader, criterion, optim, device, num_epochs=num_epochs)
    test_acc = evaluate_accuracy_mc_dropout(model, test_loader, device, T=T)
    history.append((0, test_acc))

    for step in range(num_acq_steps):
        print(f"\nAcquisition Step {step + 1}/{num_acq_steps}")

        # Acquire new points and update train and pool indices
        train_indices, pool_indices = acquire_and_add_points(
            model, T=T, pool_loader=pool_loader,
            acquisition_function=get_acquisition_function(acq_function),
            num_points=acq_size, device=device,
            train_indices=train_indices, pool_indices=pool_indices,
            bayesian=True
        )
        print(f"Acquired {acq_size} new points.")

        # Update data loaders with new indices
        train_loader, val_loader, pool_loader, _ = get_data_loaders(
            orig_trainset, testset, train_indices, val_indices, pool_indices,
            train_batch_size=16, val_batch_size=16, pool_batch_size=128, test_batch_size=256
        )
        print("Data loaders updated.")

        # Reset model, optimizer, and criterion at each acquisition step
        model = PaperCNN().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        train_model(model, train_loader, val_loader, criterion, optim, device, num_epochs=num_epochs)
        print("Model trained.")

        # Evaluate on test set and record acquired count
        test_acc = evaluate_accuracy_mc_dropout(model, test_loader, device, T=T)
        history.append(((step + 1) * acq_size, test_acc))

    return history

def run_experiment(acq_function, num_repeats=3, num_acq_steps=100, acq_size=10, num_epochs=5, T=20):
    all_histories = []
    for repeat in range(num_repeats):
        print(f"\n=== Experiment Repeat {repeat + 1}/{num_repeats} ===")
        history = run_experiment_once(
            acq_function, num_acq_steps=num_acq_steps,
            acq_size=acq_size, num_epochs=num_epochs,
            T=T
        )
        all_histories.append(history)
    return all_histories



def run_exp_deterministic_once(acq_function, num_acq_steps, acq_size, num_epochs):
    # Get MNIST data with initial split
    orig_trainset, testset, train_indices, val_indices, pool_indices = load_mnist(
        num_train_samples=20, num_val_samples=100
    )
    print("MNIST data loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_loader, val_loader, pool_loader, test_loader = get_data_loaders(
        orig_trainset, testset, train_indices, val_indices, pool_indices,
        train_batch_size=16, val_batch_size=16, pool_batch_size=128, test_batch_size=1024
    )
    print("Initial data loaders created.")

    history = [] 

    # Initial train/eval
    model = PaperCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    train_model(model, train_loader, val_loader, criterion, optim, device, num_epochs=num_epochs)
    test_acc = evaluate_accuracy_base(model, test_loader, device)
    history.append((0, test_acc))

    for step in range(num_acq_steps):
        print(f"\nAcquisition Step {step + 1}/{num_acq_steps}")

        # Acquire new points and update train and pool indices
        train_indices, pool_indices = acquire_and_add_points(
            model, T=1, pool_loader=pool_loader,
            acquisition_function=get_acquisition_function(acq_function),
            num_points=acq_size, device=device,
            train_indices=train_indices, pool_indices=pool_indices,
            bayesian=False
        )
        print(f"Acquired {acq_size} new points.")

        # Update data loaders with new indices
        train_loader, val_loader, pool_loader, _ = get_data_loaders(
            orig_trainset, testset, train_indices, val_indices, pool_indices,
            train_batch_size=16, val_batch_size=16, pool_batch_size=512, test_batch_size=1024
        )
        print("Data loaders updated.")

        # Reset model, optimizer, and criterion at each acquisition step
        model = PaperCNN().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss() 
        

        train_model(model, train_loader, val_loader, criterion, optim, device, num_epochs=num_epochs)
        print("Model trained.")

        # Evaluate on test set and record acquired count
        test_acc = evaluate_accuracy_base(model, test_loader, device)
        history.append(((step + 1) * acq_size, test_acc))
    
    return history

def run_exp_deterministic(acq_function, num_repeats=3, num_acq_steps=100, acq_size=10, num_epochs=5):
    all_histories = []
    for repeat in range(num_repeats):
        print(f"\n=== Experiment Repeat {repeat + 1}/{num_repeats} ===")
        history = run_exp_deterministic_once(
            acq_function, num_acq_steps=num_acq_steps,
            acq_size=acq_size, num_epochs=num_epochs
        )
        all_histories.append(history)

    return all_histories