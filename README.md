# PyTorch Image Classification Project

This repository contains Python scripts for training an image classification model using PyTorch. The project includes functionalities for data handling, model architecture construction, training, and evaluation.

## Overview

This project focuses on building and training a convolutional neural network (CNN) for image classification tasks using PyTorch. It consists of classes for data loading and model creation:

- **Data Handling**: Utilizes PyTorch's DataLoader to manage loading, preprocessing, and batching of image data.
- **Model Building**: Constructs a model architecture using pre-trained models like VGG13 or VGG16 and customizes the classifier for the specific task.
- **Model Training**: Trains the model using the specified hyperparameters (learning rate, epochs, etc.) and the loaded dataset.
- **Model Evaluation**: Evaluates the trained model's performance on a test dataset, computing accuracy and loss metrics.

## Usage

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/plasmatom/image_classifier.git
   cd image_classifier
   
### Running the Model Training

To train the model, run the following command:

```bash
python main.py --data_directory <path_to_data> --learning_rate 0.001 --arch vgg13 --hidden_units 512 --epochs 10 --save_dir <path_to_save_directory> --gpu



