import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import TestModel
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Define transforms for the datasets, including data augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),           # Resize images to 128x128
    # Randomly flip images horizontally
    transforms.RandomHorizontalFlip(p=0.5),
    # Randomly rotate images by +/- 15 degrees
    transforms.RandomRotation(degrees=15),
    # Randomly adjust color properties
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),                   # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                         )  # Normalize to [-1, 1]
])


# Load datasets
train_dataset = datasets.ImageFolder(
    root='fix_data/train', transform=transform)
val_dataset = datasets.ImageFolder(
    root='fix_data/validate', transform=transform)
test_dataset = datasets.ImageFolder(
    root='fix_data/test', transform=transform)  # Test dataset

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)  # Test loader

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

        # save_confusion_matrix(train_labels, train_preds, train_dataset.classes,
        #                       f'confusion_matrix_train_epoch_{epoch + 1}')

        # Calculate validation accuracy, loss, and confusion matrix
        val_accuracy, val_preds, val_labels = evaluate_model(
            model, val_loader, device)
        val_loss = criterion(model(torch.FloatTensor(val_dataset[0][0].unsqueeze(
            0).to(device))).detach(), torch.LongTensor([val_dataset[0][1]]).to(device))
        val_loss_history.append(val_loss.item())

        save_confusion_matrix(
            val_labels, val_preds, val_dataset.classes, f'confusion_matrix_val_epoch_{epoch + 1}')

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

# Function to evaluate test set and compute metrics


def evaluate_test_set(model, test_loader, device, class_names, experiment_counter, csv_file, num_epochs):
    test_accuracy, test_preds, test_labels = evaluate_model(
        model, test_loader, device)

    # Calculate precision, recall, f1 score
    precision = precision_score(test_labels, test_preds, average='macro')
    recall = recall_score(test_labels, test_preds, average='macro')
    f1 = f1_score(test_labels, test_preds, average='macro')

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(
        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Save confusion matrix for test set
    save_confusion_matrix(test_labels, test_preds, class_names,
                          f'confusion_matrix_test_{experiment_counter}')

    # Append metrics to the CSV file
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [experiment_counter, test_accuracy, precision, recall, f1, "DNN", num_epochs])

    return test_accuracy

# Function to get or initialize the experiment counter


def get_experiment_counter(csv_file):
    if not os.path.exists(csv_file):
        # Create the CSV file with headers if it doesn't exist
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Experiment', 'Test Accuracy',
                            'Precision', 'Recall', 'F1 Score', "Model", "Epoch"])
        return 1  # Start at experiment 1
    else:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            return len(rows)  # Next experiment number


# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Path for the CSV log file
csv_file = 'experiment_log.csv'

# Retrieve experiment counter from CSV file
experiment_counter = get_experiment_counter(csv_file)
num_epochs = 30
# Train the model and get the loss history
train_loss_history, val_loss_history = train(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

# Plot and save the loss history
plot_loss_history(train_loss_history, val_loss_history)

# Evaluate the model on the test set and log the metrics in CSV
test_accuracy = evaluate_test_set(
    model, test_loader, device, test_dataset.classes, experiment_counter, csv_file, num_epochs)

# Save the trained model with the accuracy and experiment counter in the filename
model_filename = f'model_dnn_exp{experiment_counter}_acc{test_accuracy:.4f}.pth'
torch.save(model.state_dict(), model_filename)
print(f'Model saved as {model_filename}')
