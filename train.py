import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import TestModel
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Define transforms for the training and validation datasets
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root='fix_data/train', transform=transform)
val_dataset = datasets.ImageFolder(
    root='fix_data/validate', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)  # Number of classes in your dataset
model = TestModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy and return predictions and labels


def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, np.array(all_preds), np.array(all_labels)

# Function to save confusion matrix


def save_confusion_matrix(true_labels, pred_labels, class_names, filename):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {filename}')
    plt.savefig(f'{filename}.png')
    plt.close()

# Training function


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    train_loss_history = []
    val_loss_history = []

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

        # Calculate training accuracy, loss, and confusion matrix
        train_accuracy, train_preds, train_labels = evaluate_model(
            model, train_loader, device)
        train_loss = running_loss / len(train_loader)
        train_loss_history.append(train_loss)

        save_confusion_matrix(train_labels, train_preds, train_dataset.classes,
                              f'confusion_matrix_train_epoch')

        # Calculate validation accuracy, loss, and confusion matrix
        val_accuracy, val_preds, val_labels = evaluate_model(
            model, val_loader, device)
        val_loss = criterion(model(torch.FloatTensor(val_dataset[0][0].unsqueeze(
            0).to(device))).detach(), torch.LongTensor([val_dataset[0][1]]).to(device))
        val_loss_history.append(val_loss.item())

        save_confusion_matrix(
            val_labels, val_preds, val_dataset.classes, f'confusion_matrix_val_epoch')

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    return train_loss_history, val_loss_history

# Function to plot and save loss history


def plot_loss_history(train_loss_history, val_loss_history):
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.savefig('loss_history.png')
    plt.close()


# Initialize the model
# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model and get the loss history
train_loss_history, val_loss_history = train(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

# Plot and save the loss history
plot_loss_history(train_loss_history, val_loss_history)

# Save the trained model
torch.save(model.state_dict(), 'model_dnn_2.pth')
print('Model saved as model_dnn.pth')
