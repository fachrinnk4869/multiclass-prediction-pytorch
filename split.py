import os
import shutil
import random


def split_dataset(original_dir, train_dir, valid_dir, test_dir, valid_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset from the original directory into training, validation, and test directories.

    Parameters:
    - original_dir: Directory containing the original dataset (class subdirectories).
    - train_dir: Directory to save the training dataset.
    - valid_dir: Directory to save the validation dataset.
    - test_dir: Directory to save the test dataset.
    - valid_ratio: Fraction of data to use for validation.
    - test_ratio: Fraction of data to use for testing.
    """
    listdir = os.listdir(original_dir)
    # Create training, validation, and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop through each class folder in the original directory
    for class_name in listdir:
        class_path = os.path.join(original_dir, class_name)

        # Check if it's a directory (to ensure it's a class folder)
        if os.path.isdir(class_path):
            # Create class folders in train, valid, and test directories
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Get list of all images in the class folder
            images = os.listdir(class_path)
            random.shuffle(images)  # Shuffle the images for randomness

            # Calculate the number of images for validation and test sets
            num_total = len(images)
            num_test = int(num_total * test_ratio)
            num_valid = int(num_total * valid_ratio)
            num_train = num_total - num_valid - num_test

            # Split the images into training, validation, and test sets
            test_images = images[:num_test]
            valid_images = images[num_test:num_test + num_valid]
            train_images = images[num_test + num_valid:]

            # Move images to their respective folders
            for image in train_images:
                shutil.copy(os.path.join(class_path, image),
                            os.path.join(train_dir, class_name, image))
            for image in valid_images:
                shutil.copy(os.path.join(class_path, image),
                            os.path.join(valid_dir, class_name, image))
            for image in test_images:
                shutil.copy(os.path.join(class_path, image),
                            os.path.join(test_dir, class_name, image))

    print("Dataset has been split into training, validation, and test sets.")


# Example usage

# Update with your original dataset path
original_dataset_dir = 'fix_data'
# Desired path for training dataset
train_dataset_dir = os.path.join(original_dataset_dir, 'train')
# Desired path for validation dataset
valid_dataset_dir = os.path.join(original_dataset_dir, 'validate')
# Desired path for test dataset
test_dataset_dir = os.path.join(original_dataset_dir, 'test')

split_dataset(original_dataset_dir, train_dataset_dir,
              valid_dataset_dir, test_dataset_dir, valid_ratio=0.2, test_ratio=0.1)
