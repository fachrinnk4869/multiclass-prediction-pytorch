import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Define the transformations for training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),   # Resize images
        transforms.RandomHorizontalFlip(),  # Randomly flip images
        transforms.ToTensor(),
        # Normalize for pre-trained models
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets from folders
data_dir = 'data'  # Update with the path to your dataset root folder
image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
                  for x in ['train', 'validate']}

# Create data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'validate']}

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained model (ResNet18)
model = models.resnet18(pretrained=True)

# Modify the fully connected layer to match our number of classes (3: apple, papaya, banana)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 classes

# Move model to device (GPU if available)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Softmax layer for output during evaluation
softmax = nn.Softmax(dim=1)

# Training function


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # Apply softmax during evaluation
                    if phase == 'validate':
                        # Apply softmax for validation predictions
                        outputs = softmax(outputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model


# Train the model
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

# Save the trained model
torch.save(model.state_dict(), 'food_classifier.pth')
print('Model saved as food_classifier.pth')
