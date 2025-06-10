# Machine Learning Projects 

This repository is dedicated to projects completed as part of the **Machine Learning with Python** certification from [freeCodeCamp](https://www.freecodecamp.org/).  
Each subfolder will contain a separate project. This is the first project: **Healthcare Cost Prediction**.

---

## Project 1: Healthcare Cost Prediction Challenge

### Overview

In this challenge, the goal is to predict healthcare costs using a regression algorithm.  
You are provided with a dataset containing information about various individuals, including their healthcare expenses. The objective is to build a machine learning model that accurately predicts healthcare costs for new cases.

---

### Data Preparation Steps

- **Categorical Data Handling:**  
  All categorical data is converted to numerical format to ensure compatibility with regression algorithms.

- **Train/Test Split:**  
  The dataset is split into two sets:
  - **80% for training** (`train_dataset`)
  - **20% for testing** (`test_dataset`)

- **Label Extraction:**  
  The `"expenses"` column (the prediction target) is popped off from both datasets to create:
  - `train_labels`
  - `test_labels`
  These labels are used during model training and evaluation.

---

### Model Building & Training

- **Library:**  
  The project uses **Keras** to manipulate data, build the regression model, and perform training.

- **Model Training:**  
  The model is trained with the `train_dataset` and corresponding `train_labels`.

---

### Model Evaluation

- The final cell in the notebook evaluates the model using the **unseen** `test_dataset`.
- **Success Criterion:**  
  To pass the challenge, `model.evaluate()` must return a **Mean Absolute Error (MAE) under 3500**. This means the model predicts healthcare costs within $3,500 of the actual value.
- The final cell also predicts expenses for the test data and visualizes the results in a graph.

---

## How to Run

- This repository is intended to be opened and run in **Google Colab** for the best experience.
- Make sure to execute the cells in order, and review the final cell to check the model's performance and predictions.

---

## Projects List

1. **Healthcare Cost Prediction** (this folder)

More projects will be added to this repository as the certification progresses.

---

**Happy Learning!**
