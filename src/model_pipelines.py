import torch
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    """Train the model."""
    model.train()
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
    return train_losses, val_losses


def evaluate_loss(model, data_loader, criterion, device):
    """Evaluate the model loss. Just for monitoring training progress."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    total_loss = running_loss / len(data_loader.dataset)
    return total_loss


def evaluate_accuracy_base(model, data_loader, device):
    """Evaluate the model accuracy on the given data loader with the classic method ."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)    # not using softmax as it does not change the argmax
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def evaluate_accuracy_mc_dropout(model, data_loader, device, T=20):
    """Evaluate the model accuracy on the given data loader using MC Dropout with T forward passes."""
    model.train()    # Keep dropout active
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            B = inputs.size(0)
            # Repeat batch T times for vectorized MC dropout
            inputs_repeat = inputs.repeat((T, 1, 1, 1))  # (T*B, ...)
            outputs = model(inputs_repeat)
            probs = F.softmax(outputs, dim=1)
            outputs_T = probs.view(T, B, -1)  # Shape: (T, B, C)
            avg_outputs = outputs_T.mean(dim=0)  # Average over T

            _, predicted = torch.max(avg_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy