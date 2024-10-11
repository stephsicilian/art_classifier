# Art Classification with Convolutional Neural Networks (CNN)

This project is a binary image classifier built with PyTorch, designed to distinguish between two types of art: **sculptures** and **paintings**. The model uses a Convolutional Neural Network (CNN) architecture to classify images and is trained on a dataset of artwork images. <br>

## Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Requirements](#requirements)  
- [Setup and Installation](#setup-and-installation)  
- [Training the Model](#training-the-model)  
- [Testing the Model](#testing-the-model)  
- [Results](#results)  
- [File Structure](#file-structure)  
- [License](#license)  
<br>

## Project Overview

This project implements a CNN model to classify artworks as either sculptures or paintings. The project uses PyTorch as the deep learning framework and includes scripts for preprocessing, training, and testing the model. <br>

The key objectives of this project are:  
- To preprocess images for input into the CNN.  
- To build and train a CNN to classify images.  
- To test the trained model on a separate test set and evaluate its accuracy.  
<br>

## Dataset

The dataset used for this project contains images of:  
- **Sculptures**  
- **Paintings**  

The dataset is divided into training and test sets and follows the directory structure:  <br>
datasets/ ├── train/ │ ├── sculpture/ │ └── painting/ └── test/ ├── sculpture/ └── painting/
<br>


Each subdirectory contains image files in `.jpg` or `.png` format. <br>  
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving).  
<br>

## Model Architecture

The neural network is a simple CNN with the following layers:  
- 2 convolutional layers with ReLU activation functions  
- 1 batch normalization layer  
- 1 fully connected layer  
- Sigmoid activation function for binary classification  

The network outputs a single probability value representing whether the image is classified as a painting (`1`) or a sculpture (`0`).  
<br>

## Requirements

To run this project, you need the following dependencies:  
- Python 3.x  
- PyTorch  
- NumPy  
- Pillow (PIL)  
You can install the necessary dependencies by running:  
```bash
pip install torch numpy pillow

## Setup and Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/art-classification.git  
   cd art-classification

