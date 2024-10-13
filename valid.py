import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import TestModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define transforms for the training and validation datasets
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load validation dataset
val_dataset = datasets.ImageFolder(
    root='validate', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Function to validate the model


# Define evaluation function
def validate(model, test_loader, device, output_file="confusion_matrix.png"):
    model.eval()  # Set the model to evaluation mode (disable dropout, batch norm, etc.)
    correct = 0
    total = 0
    all_preds = []  # Store all predictions
    all_labels = []  # Store all ground truth labels

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # Forward pass
            # Get the index of the max log-probability (predicted class)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            # Count correct predictions
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    # Save the confusion matrix as an image file
    plt.savefig(output_file)  # Save to the specified file
    print(f"Confusion matrix saved as {output_file}")
    # Print validation accuracy
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Load the model checkpoint


def load_model(model_path, num_classes, device):
    # Initialize the model
    # Ensure this matches your saved model's architecture
    model = TestModel(num_classes)
    model.to(device)  # Move the model to the appropriate device

    # Load model weights from the saved file
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model and load the checkpoint
num_classes = len(val_dataset.classes)  # Number of classes in your dataset
model_path = 'model.pth'  # Path to the saved model checkpoint
model = load_model(model_path, num_classes, device)

# Validate the model
validate(model, val_loader, device)
