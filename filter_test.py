import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import os
import shutil

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transform (same as during training)
data_transforms = transforms.Compose([
    # Resize images to match training dimensions
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
model = models.resnet18(pretrained=False)  # Initialize ResNet18
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # Adjust for 3 classes
model.load_state_dict(torch.load('food_classifier.pth')
                      )  # Load trained weights
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define class names
class_names = ['apel', 'pepaya', 'pisang']

# Define the confidence threshold
threshold = 0.8

# Folder paths
# Path to the folder containing test images
test_images_dir = 'dataset/apel'
output_dir = 'sorted_results'  # Folder to store sorted images

# Create folders for each class and unlabeled
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'unlabeled'), exist_ok=True)

# Function to predict the class of an image


def predict_image(image_path, model, threshold):
    image = Image.open(image_path).convert('RGB')  # Open image
    # Apply transformations and add batch dimension
    image = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():  # No gradient calculation
        outputs = model(image)  # Forward pass
        probabilities = nn.Softmax(dim=1)(outputs)  # Get softmax probabilities
        # Get max probability and class
        confidence, preds = torch.max(probabilities, 1)

        # If confidence is above the threshold, classify to that class
        if confidence.item() >= threshold:
            return class_names[preds.item()], confidence.item()
        else:
            return 'unlabeled', confidence.item()


# Iterate over test images and classify them
for image_file in os.listdir(test_images_dir):
    if image_file.endswith(('jpg', 'jpeg', 'png')):
        image_path = os.path.join(test_images_dir, image_file)

        # Predict the class
        predicted_class, confidence = predict_image(
            image_path, model, threshold)
        print(
            f'Image: {image_file} | Predicted: {predicted_class} | Confidence: {confidence:.4f}')

        # Move the image to the corresponding folder
        output_class_dir = os.path.join(output_dir, predicted_class)
        shutil.copy(image_path, os.path.join(output_class_dir, image_file))

print(
    f"Images sorted into respective folders based on a threshold of {threshold}.")
