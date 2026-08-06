# 🧠 CNN Image Classification using TensorFlow

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

# 📖 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to perform image classification on the **CIFAR-10 dataset**.

The project demonstrates the complete Deep Learning workflow—from loading and preprocessing image data to building, training, evaluating, and saving a CNN model.

Developed as part of an **AI Engineering Internship**, this project provides practical experience in implementing deep learning models for image classification using modern AI frameworks.

The project follows a modular and object-oriented design, making it easy to understand, maintain, and extend.

---

# 🎯 Objectives

The main objectives of this project are:

- Learn the fundamentals of Convolutional Neural Networks.
- Understand image preprocessing techniques.
- Build a CNN architecture from scratch.
- Train a Deep Learning model using TensorFlow/Keras.
- Evaluate model performance on unseen data.
- Save trained models for future predictions.
- Visualize model performance using graphs.
- Gain hands-on experience with Deep Learning.

---

# ✨ Project Features

✔ Object-Oriented Programming Structure

✔ Modular Code Design

✔ Automatic CIFAR-10 Dataset Loading

✔ Image Normalization

✔ CNN Built from Scratch

✔ TensorFlow/Keras Implementation

✔ Automatic Model Training

✔ Model Evaluation

✔ Classification Report

✔ Model Saving

✔ Performance Visualization

✔ Easy-to-Understand Code

✔ Beginner Friendly

---

# 🖼 Dataset Information

## Dataset

**CIFAR-10**

The CIFAR-10 dataset is one of the most popular benchmark datasets used for image classification.

It contains **60,000 RGB images** belonging to **10 different classes**.

---

## Dataset Statistics

| Property | Value |
|-----------|-------|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 × 32 |
| Color Channels | 3 (RGB) |
| Number of Classes | 10 |

---

## Dataset Classes

```text
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
```

---

# 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | High-Level Neural Network API |
| NumPy | Numerical Computation |
| Pandas | Data Processing |
| Matplotlib | Data Visualization |
| Scikit-Learn | Performance Metrics |
| OpenCV | Image Processing |

---

# 🎓 Skills Demonstrated

This project demonstrates practical knowledge of:

- Deep Learning
- Convolutional Neural Networks (CNN)
- TensorFlow
- Keras
- Image Classification
- Image Preprocessing
- Feature Extraction
- Model Evaluation
- Data Visualization
- Object-Oriented Programming
- Python Programming

---

# 🧠 About CNN

A **Convolutional Neural Network (CNN)** is a specialized type of Artificial Neural Network designed to process image data.

Unlike traditional neural networks, CNNs automatically learn image features such as:

- Edges
- Shapes
- Colors
- Textures
- Patterns
- Objects

This makes CNNs highly effective for computer vision tasks including:

- Image Classification
- Face Recognition
- Medical Image Analysis
- Object Detection
- Autonomous Driving
- Handwritten Digit Recognition

---

# 🎯 Project Goal

The primary goal of this project is **learning**, not achieving state-of-the-art accuracy.

This implementation focuses on understanding the complete Deep Learning pipeline, including:

- Loading datasets
- Image preprocessing
- CNN architecture design
- Model compilation
- Training
- Evaluation
- Prediction
- Model persistence

The project serves as a strong foundation for developing more advanced CNN models in future projects.

---
# 🏗 Project Structure

The project follows a clean and modular object-oriented structure for better readability and maintainability.

```text
Task_2_Intermediate
│
├── datasets
│
├── models
│   └── CNN
│       └── cnn_model.keras
│
├── outputs
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   └── prediction.png
│
├── report
│
├── src
│   └── CNN
│       ├── cnn_main.py
│       ├── CNN_Model.py
│       ├── CNN_Training.py
│       ├── CNN_Evaluation.py
│       └── README.md
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 CNN Architecture

The model consists of two convolutional blocks followed by fully connected layers.

```text
Input Image
(32 × 32 × 3)

        │

        ▼

Conv2D
32 Filters
3×3 Kernel
ReLU

        │

        ▼

MaxPooling2D
2×2

        │

        ▼

Conv2D
64 Filters
3×3 Kernel
ReLU

        │

        ▼

MaxPooling2D
2×2

        │

        ▼

Flatten

        │

        ▼

Dense
128 Neurons
ReLU

        │

        ▼

Dropout
0.5

        │

        ▼

Dense
10 Neurons
Softmax

        │

        ▼

Predicted Class
```

---

# 🏛 Model Architecture Summary

| Layer | Purpose |
|---------|---------|
| Input Layer | Accept RGB Images |
| Conv2D (32 Filters) | Feature Extraction |
| MaxPooling2D | Down Sampling |
| Conv2D (64 Filters) | Learn Deeper Features |
| MaxPooling2D | Reduce Spatial Dimensions |
| Flatten | Convert Feature Maps into Vector |
| Dense (128) | Learn High-Level Features |
| Dropout (0.5) | Reduce Overfitting |
| Dense (10) | Multi-Class Classification |

---

# 🔄 Project Workflow

The complete workflow of the project is illustrated below.

```text
Load Dataset

      │

      ▼

Preprocess Images

      │

      ▼

Normalize Images

      │

      ▼

Build CNN Model

      │

      ▼

Compile Model

      │

      ▼

Train Model

      │

      ▼

Validate Model

      │

      ▼

Evaluate Model

      │

      ▼

Predict Classes

      │

      ▼

Save Model

      │

      ▼

Generate Output Graphs
```

---

# 📂 File Description

## 📌 cnn_main.py

The main entry point of the application.

Responsibilities:

- Starts the project
- Calls the training pipeline
- Calls the evaluation pipeline
- Displays project results

---

## 📌 CNN_Model.py

Responsible for creating the CNN architecture.

Contains:

- Conv2D Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Dropout Layer
- Output Layer

---

## 📌 CNN_Training.py

Responsible for:

- Loading the CIFAR-10 dataset
- Data preprocessing
- Image normalization
- Model compilation
- Model training
- Saving trained model

---

## 📌 CNN_Evaluation.py

Responsible for:

- Model evaluation
- Accuracy calculation
- Predictions
- Classification report
- Confusion matrix generation
- Performance visualization

---

# ⚙ Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Activation Function | ReLU |
| Output Activation | Softmax |
| Epochs | 10 |
| Batch Size | 32 |
| Dropout Rate | 0.5 |
| Image Size | 32 × 32 × 3 |

---

# 🧮 Deep Learning Pipeline

```text
Input Image

↓

Image Preprocessing

↓

Normalization

↓

Convolution

↓

Feature Maps

↓

Activation (ReLU)

↓

Pooling

↓

Flatten

↓

Dense Layer

↓

Softmax Output

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Weight Update

↓

Improved Prediction
```

---

# 🔍 Data Preprocessing

Before training the CNN model, the dataset undergoes preprocessing to improve model performance.

The preprocessing steps include:

- Loading the CIFAR-10 dataset
- Splitting into training and testing datasets
- Converting pixel values from integers to floating-point values
- Normalizing pixel values from **0–255** to **0–1**
- Preparing data for CNN input

These preprocessing steps help the model converge faster and improve learning stability.

---

# 🧠 Why CNN?

Traditional Artificial Neural Networks (ANNs) are not efficient for image processing because they require a large number of parameters.

CNNs solve this problem by:

- Automatically extracting important image features
- Reducing the number of trainable parameters
- Learning spatial relationships
- Improving classification performance
- Preserving important visual patterns

This makes CNNs the preferred choice for computer vision applications.

---

# 📊 CNN Learning Process

During training, the CNN repeatedly performs the following sequence:

```text
Input Image

↓

Convolution

↓

Activation (ReLU)

↓

Pooling

↓

Flatten

↓

Dense Layer

↓

Prediction

↓

Loss Calculation

↓

Backpropagation

↓

Update Weights

↓

Repeat for Next Epoch
```

---

# 🎯 Advantages of This Project

- Clean Object-Oriented Design
- Easy to Understand
- Modular Implementation
- Beginner Friendly
- Well Documented
- Reusable Components
- Easy to Extend
- Suitable for Academic Projects
- Suitable for AI Internship Submission
- Strong Foundation for Advanced CNN Models

---
# 📊 Training Process

The model is trained using the **Adam Optimizer** and **Sparse Categorical Crossentropy** loss function for **10 epochs** with a batch size of **32**.

During each epoch, the CNN performs the following operations:

1. Forward Propagation
2. Feature Extraction
3. Loss Calculation
4. Backpropagation
5. Weight Updates
6. Validation on Test Data

The training process continues until all epochs are completed, allowing the model to gradually improve its ability to classify images.

---

# 📈 Training Workflow

```text
Dataset Loaded

↓

Images Normalized

↓

CNN Model Created

↓

Model Compiled

↓

Training Started

↓

Epoch 1

↓

Epoch 2

↓

...

↓

Epoch 10

↓

Training Completed

↓

Model Saved
```

---

# 📊 Model Performance

After training the CNN model on the CIFAR-10 dataset, the following performance was achieved.

| Metric | Value |
|----------|---------|
| Dataset | CIFAR-10 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Epochs | 10 |
| Batch Size | 32 |
| Optimizer | Adam |
| Test Accuracy | **70.74%** |
| Test Loss | **0.8553** |

---

# 📈 Performance Analysis

The model successfully learned meaningful image features and achieved approximately **70.74% classification accuracy** on unseen test images.

Although this is not the highest possible accuracy for the CIFAR-10 dataset, it demonstrates that the implemented CNN architecture is capable of learning representative features and performing multi-class image classification effectively.

This project primarily focuses on understanding the complete CNN implementation rather than maximizing benchmark performance.

---

# 📊 Classification Performance

The trained model was evaluated using Precision, Recall, and F1-Score for each class.

| Class | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Airplane | 0.74 | 0.76 | 0.75 |
| Automobile | 0.77 | 0.88 | 0.82 |
| Bird | 0.62 | 0.58 | 0.60 |
| Cat | 0.51 | 0.54 | 0.53 |
| Deer | 0.71 | 0.56 | 0.62 |
| Dog | 0.55 | 0.68 | 0.61 |
| Frog | 0.77 | 0.77 | 0.77 |
| Horse | 0.82 | 0.71 | 0.76 |
| Ship | 0.81 | 0.83 | 0.82 |
| Truck | 0.81 | 0.77 | 0.79 |

These metrics provide a more detailed understanding of the model's performance across different image categories.

---

# 📂 Generated Output Files

After successful execution, the project automatically generates the following outputs.

```text
models/
└── CNN
    └── cnn_model.keras

outputs/
├── accuracy.png
├── loss.png
├── confusion_matrix.png
├── prediction.png
└── classification_report.txt
```
---

# 📈 Output Visualizations

The project generates graphical visualizations that help analyze model performance.

These include:

- 📊 Training Accuracy Curve
- 📉 Training Loss Curve
- 📋 Classification Report
- 🔲 Confusion Matrix
- 🖼 Sample Predictions
- 📈 Evaluation Metrics

These outputs are stored inside the **outputs/** directory for future reference.

---

# 📷 Sample Outputs

Example generated files:

```text
outputs/

├── training_accuracy.png

├── training_loss.png

├── confusion_matrix.png

├── classification_report.txt

└── prediction.png
```

These visualizations make it easier to understand the model's learning behavior and identify areas for improvement.

---

# 💻 Example Console Output

```text
===================================================
        CNN IMAGE CLASSIFICATION PROJECT
===================================================

Loading CIFAR-10 Dataset...

Dataset Loaded Successfully

Training Images : (50000, 32, 32, 3)

Testing Images  : (10000, 32, 32, 3)

Preprocessing Images...

Images Normalized Successfully

Building CNN Model...

CNN Model Created Successfully

Compiling Model...

Training Started...

Epoch 1/10
...

Epoch 10/10

Training Completed Successfully

Saving Model...

Model Saved Successfully

Evaluating Model...

Test Accuracy : 70.74%

Test Loss : 0.8553

Classification Report Generated Successfully

Confusion Matrix Generated Successfully

Project Completed Successfully
```

---

# 📌 Current Performance & Limitations

The current CNN model achieves a **Test Accuracy of approximately 70.74%** on the CIFAR-10 dataset.

While this demonstrates that the model successfully learns important image features and performs multi-class classification, there is still considerable room for improvement.

The primary objective of this project was **to understand and implement the complete Deep Learning pipeline from scratch**, including:

- Dataset Loading
- Image Preprocessing
- CNN Architecture Design
- Model Compilation
- Model Training
- Performance Evaluation
- Prediction
- Model Saving

rather than achieving state-of-the-art benchmark accuracy.

This implementation serves as a **baseline CNN model**, providing a strong foundation for experimenting with more advanced architectures and optimization techniques.

---

# 🚀 Future Optimization Plan

The current implementation can be further improved by incorporating advanced deep learning techniques.

Future improvements include:

- Data Augmentation
- Batch Normalization
- Transfer Learning
- Early Stopping
- Learning Rate Scheduler
- Model Checkpointing
- Hyperparameter Tuning
- More Convolution Layers
- Residual Connections
- TensorBoard Integration
- Better Regularization Techniques
- Experimenting with Different Optimizers

These enhancements are expected to improve feature extraction, reduce overfitting, increase generalization, and significantly improve the overall classification accuracy.

The long-term goal is to achieve **higher accuracy while maintaining a clean, modular, and scalable codebase suitable for real-world image classification applications.**

---