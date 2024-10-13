import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import TestModel
from sklearn.metrics import accuracy_score

# Define transforms for the training and validation datasets
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load datasets (replace 'path_to_train_data' and 'path_to_valid_data' with your dataset paths)
train_dataset = datasets.ImageFolder(
    root='train', transform=transform)
val_dataset = datasets.ImageFolder(
    root='validate', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)  # Number of classes in your dataset
model = TestModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy


def calculate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Training function


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        print(f"Starting Epoch {epoch + 1}/{num_epochs}...")

        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            print(f"Batch {i}: Loss={loss.item():.4f}")
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Calculate training accuracy
        train_accuracy = calculate_accuracy(model, train_loader, device)
        # Calculate validation accuracy
        val_accuracy = calculate_accuracy(model, val_loader, device)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')


# Initialize the model
# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model and check accuracy at each epoch
train(model, train_loader, val_loader, criterion,
      optimizer, num_epochs=10, device=device)

# Save the model
torch.save(model.state_dict(), 'model.pth')
print('Model saved as model.pth')
