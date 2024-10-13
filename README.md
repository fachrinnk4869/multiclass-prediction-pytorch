# Image Multi Class Classification

This project implements an image classification. It includes scripts for training, testing, auto-labeling images and image scrappers. The project is containerized using Docker and Docker Compose for easy setup and deployment.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Docker](#docker)
- [Current Accuracy](#current-accuracy)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fachrinnk4869/multiclass-prediction-pytorch
   cd multiclass-prediction-pytorch
   ```

2. Build the Docker image:
   ```bash
   docker compose build
   ```

3. Run the Docker container:
   ```bash
   docker compose up
   ```

## Usage

### Split Dataset

First of all, we have to split dataset into train and validate (configurations are in the file), run:
```bash
python split.py
```

### Training the Model

To train the model, run:
```bash
python train.py
```

### Testing the Model

To test the model, run:
```bash
python test.py
```

### Auto-Labeling Images

Train images using the trained model with small data, run:
```bash
python filter_train.py 
```

To automatically label images using the trained model, run:
```bash
python filter_test.py 
```

## Current Accuracy

The currently achieved accuracy of the model is **Epoch [10/10], Train Accuracy: 0.9765, Validation Accuracy: 0.6887**.

## File Structure

```
multiclass-prediction-pytorch/
│
├── dataset/                    # Directory containing training and testing images(raw data from scrapping)
│   ├── train/                  # Training images organized by class
│   └── validate/               # Validation images organized by class
├── data_fix/                   # Filtered data
├── data/                       # Small data to support auto labelling
│
├── model.py                    # Model definition
├── image_scrapper.ipnb         # Scrapping desired images
├── train.py                    # Script to train the model
├── test.py                     # Script to test the model
├── filter.py                   # Script for auto-labeling images
├── requirements.txt            # Required Python packages
├── Dockerfile                  # Dockerfile for the PyTorch environment
├── docker-compose.yml          # Docker Compose configuration
└── README.md                   # Project documentation
```

## Docker

This project includes a Dockerfile and `docker-compose.yml` for easy deployment. 

### Dockerfile

The Dockerfile sets up a PyTorch environment with the necessary dependencies. You can customize it further based on your needs.

### docker-compose.yml

The Docker Compose configuration helps you manage multi-container applications. It is designed to run the application seamlessly in a containerized environment.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```