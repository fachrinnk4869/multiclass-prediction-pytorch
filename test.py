import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import shutil
from model import TestModel  # Make sure to import your custom model

# Set up device for model inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms (same as during training)
data_transforms = transforms.Compose([
    # Resize images to match training dimensions
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Adjust normalization based on your training data
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the trained model
num_classes = 3  # Update this to match your model's number of output classes
model = TestModel(num_classes)  # Initialize your TestModel
model.load_state_dict(torch.load('model_dnn_fix.pth'))  # Load trained weights
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define class names
class_names = ['apel', 'pepaya', 'pisang']  # Update according to your classes

# Define the confidence threshold
threshold = 0.8  # You can adjust this value based on your requirements

# Folder paths
# Path to the folder containing test images
test_images_dir = 'fix_data/apel_fix'
output_dir = 'results'  # Folder to store sorted images

# Create output directories for classes and unlabeled
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'unlabeled'), exist_ok=True)

# Function to predict the class of an image


def predict_image(image_path, model, threshold):
    image = Image.open(image_path).convert(
        'RGB')  # Open the image and convert to RGB
    # Apply transformations and add batch dimension
    image = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)  # Forward pass
        probabilities = nn.Softmax(dim=1)(outputs)  # Get softmax probabilities
        # Get max probability and class
        confidence, preds = torch.max(probabilities, 1)

        # Classify based on threshold
        if confidence.item() >= threshold:
            return class_names[preds.item()], confidence.item()
        else:
            return 'unlabeled', confidence.item()


# Iterate over test images and classify them
for image_file in os.listdir(test_images_dir):
    if image_file.lower().endswith(('jpg', 'jpeg', 'png')):  # Check for image file types
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
