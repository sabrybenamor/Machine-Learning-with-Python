# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

TSV_COLUMN_NAMES = ["Nature", "Message core"]
train_df = pd.read_csv(train_file_path, names=TSV_COLUMN_NAMES, sep='\t')
test_df = pd.read_csv(test_file_path, names=TSV_COLUMN_NAMES, sep='\t')


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["Nature"].values)
y_test = label_encoder.transform(test_df["Nature"].values)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["Message core"].values).toarray()
X_test = vectorizer.transform(test_df["Message core"].values).toarray()

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

def prepare_input(messages):
    """
    Transforms a list or array of raw text messages into numerical vectors
    using the globally available fitted TF-IDF vectorizer.

    Args:
        messages (list or np.array): The raw SMS messages (strings).

    Returns:
        np.array: The TF-IDF-transformed input suitable for model.predict or model.evaluate.
    """
    return vectorizer.transform(messages).toarray()

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    prediction = []
    pred_input = prepare_input([pred_text])
    predictions = model.predict(pred_input)
    # Cast to regular Python float
    result = float(predictions[0][0] if predictions.ndim == 2 else predictions[0])
    if result < 0.5:
        prediction.append(result)
        prediction.append("ham")
    else:
        prediction.append(result)
        prediction.append("spam")
    return prediction

# Example usage:
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
