# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

df=pd.get_dummies(dataset,drop_first=True)
from sklearn.model_selection import train_test_split
train_dataset,test_dataset=train_test_split(df,test_size=0.2,random_state=0)
train_labels=train_dataset.pop('expenses')
test_labels=test_dataset.pop('expenses')

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[train_dataset.shape[1]]),  # First hidden layer
    layers.Dense(32, activation='relu'),                                         # Second hidden layer
    layers.Dense(16, activation='relu'),                                         # Third hidden layer
    layers.Dense(1)  # Output layer for regression (no activation)
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae','mse']
)

# Train the model
model.fit(train_dataset, train_labels, epochs=100, verbose=1)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
