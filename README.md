# Machine Learning Projects

This repository contains projects completed as part of the **Machine Learning with Python** certification from [freeCodeCamp](https://www.freecodecamp.org/).  
Each subfolder is a standalone project. Below are the included projects:

---

## Project 1: Healthcare Cost Prediction Challenge

### Overview

Predict healthcare costs using a regression model.  
A dataset with various individual information and healthcare expenses is provided. The goal is to build a machine learning model that accurately predicts healthcare costs for new data.

### Data Preparation

- **Categorical Data Handling:**  
  All categorical variables are encoded numerically to ensure compatibility with regression algorithms.

- **Train/Test Split:**  
  The dataset is divided into training (80%) and testing (20%) subsets.

- **Label Extraction:**  
  The target variable, `"expenses"`, is separated from the features for both the training and testing sets.

### Model Building & Training

- **Library:**  
  Utilizes **Keras** for data manipulation, model construction, and training.

- **Model Training:**  
  The model is trained using the processed training data and corresponding labels.

### Model Evaluation

- The model is evaluated on unseen test data.
- **Success Criterion:**  
  Achieving a **Mean Absolute Error (MAE) under 3500** with `model.evaluate()` is required for passing. This means predictions are within $3,500 of the true expense values.
- Predictions for the test set are visualized in a final plot.

---

## Project 2: SMS Spam Classifier

### Overview

Develop a machine learning model to classify SMS messages as either "ham" (legitimate) or "spam" (unsolicited/advertising).

### Task Details

- **Function Creation:**  
  Implements a `predict_message` function that accepts a message string and returns:
  - A probability (between 0 and 1) indicating the likelihood of "ham" (0) or "spam" (1).
  - The predicted label: "ham" or "spam".

- **Dataset:**  
  Uses the SMS Spam Collection dataset, with predefined training and test splits.

- **Notebook Structure:**  
  Libraries and data are loaded at the beginning, and model testing occurs at the end. Solution code is written in the intermediate cells.

---

## Project 3: Cats vs Dogs Image Classifier

### Overview

Build a convolutional neural network using **TensorFlow 2.0** and **Keras** to classify images as cats or dogs, with a target accuracy of at least 63% (extra credit for 70%+).

### Dataset Structure

After downloading and extracting, the data is organized as follows:

```
cats_and_dogs
├── train
│   ├── cats: [cat.0.jpg, ...]
│   └── dogs: [dog.0.jpg, ...]
├── validation
│   ├── cats: [cat.2000.jpg, ...]
│   └── dogs: [dog.2000.jpg, ...]
└── test: [1.jpg, 2.jpg, ...]
```
- The `test` directory contains unlabeled images.

### Workflow Summary

- **Data Preparation:**  
  - Set up image generators for training, validation, and test datasets using `ImageDataGenerator`.
  - Images are rescaled from 0-255 to 0-1 floating point tensors.
  - For the test set, shuffling is disabled to preserve order.

- **Data Augmentation:**  
  - The training generator applies random image transformations to help reduce overfitting.

- **Model Construction:**  
  - A Keras Sequential model is built with Conv2D, MaxPooling2D, and fully connected layers.
  - Compiled with appropriate optimizer, loss, and accuracy metrics.

- **Training:**  
  - The model is trained on the processed data, and training/validation metrics are tracked.

- **Evaluation and Prediction:**  
  - Model performance is visualized.
  - The model predicts for each test image whether it is a cat or a dog, outputting probabilities as integer labels.
  - Results are plotted for visual inspection.

- **Goal:**  
  Achieve at least 63% accuracy on classifying the test images.

---

## How to Run

- These projects are best run in **Google Colab** for ease of use.
- Follow the instructions in each notebook and execute the cells in order.
- Review the final outputs and plots to assess model performance.

---

**Happy Learning!**
