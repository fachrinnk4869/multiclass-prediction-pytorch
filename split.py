import os
import shutil
import random


def split_dataset(original_dir, train_dir, valid_dir, valid_ratio=0.2):
    """
    Splits the dataset from the original directory into training and validation directories.

    Parameters:
    - original_dir: Directory containing the original dataset (class subdirectories).
    - train_dir: Directory to save the training dataset.
    - valid_dir: Directory to save the validation dataset.
    - valid_ratio: Fraction of data to use for validation.
    """
    listdir = os.listdir(original_dir)
    # Create training and validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Loop through each class folder in the original directory
    for class_name in listdir:
        class_path = os.path.join(original_dir, class_name)

        # Check if it's a directory (to ensure it's a class folder)
        if os.path.isdir(class_path):
            # Create class folders in train and valid directories
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

            # Get list of all images in the class folder
            images = os.listdir(class_path)
            random.shuffle(images)  # Shuffle the images for randomness

            # Split the images into training and validation sets
            num_valid = int(len(images) * valid_ratio)
            valid_images = images[:num_valid]
            train_images = images[num_valid:]

            # Move images to their respective folders
            for image in train_images:
                shutil.copy(os.path.join(class_path, image),
                            os.path.join(train_dir, class_name, image))
            for image in valid_images:
                shutil.copy(os.path.join(class_path, image),
                            os.path.join(valid_dir, class_name, image))

    print("Dataset has been split into training and validation sets.")


# Example usage

# Update with your original dataset path
original_dataset_dir = 'fix_data'
# Desired path for training dataset
train_dataset_dir = os.path.join(original_dataset_dir, 'train')
# Desired path for validation dataset
valid_dataset_dir = os.path.join(original_dataset_dir, 'validate')

split_dataset(original_dataset_dir, train_dataset_dir,
              valid_dataset_dir, valid_ratio=0.2)
