# Machine Learning Projects 

This repository is dedicated to projects completed as part of the **Machine Learning with Python** certification from [freeCodeCamp](https://www.freecodecamp.org/).  
Each subfolder contains a separate project. Below are the listed projects:

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

## Project 2: SMS Spam Classifier

### Overview

In this challenge, you need to create a machine learning model that will classify SMS messages as either "ham" or "spam".  
A "ham" message is a normal message sent by a friend. A "spam" message is an advertisement or a message sent by a company.

---

### Task Details

- **Function Creation:**  
  You should create a function called `predict_message` that takes a message string as an argument and returns a list. 
  - The first element in the list should be a number between zero and one that indicates the likeliness of "ham" (0) or "spam" (1).
  - The second element in the list should be the word "ham" or "spam", depending on which is most likely.

- **Dataset:**  
  The SMS Spam Collection dataset is used for this project. The dataset has already been grouped into train data and test data.

- **Notebook Structure:**  
  - The first two cells import the libraries and data.
  - The final cell tests your model and function.
  - Add your code in between these cells.

---

## How to Run

- This repository is intended to be opened and run in **Google Colab** for the best experience.
- Make sure to execute the cells in order, and review the final cell to check the model's performance and predictions.

---


**Happy Learning!**
