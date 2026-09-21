"""
MNIST Classifier with Nadam Optimizer
-------------------------------------
Trains a fully connected neural network on MNIST using the Nadam optimizer
(the best performer from optimizer_comparison.py) and prints the final
loss and accuracy in a table.
"""

import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Nadam


# ---------------------------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten 28 x 28 images into 784-length vectors
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255


# ---------------------------------------------------------------------------
# 2. Model definition
# ---------------------------------------------------------------------------
def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation="softmax"))

    return model


# ---------------------------------------------------------------------------
# 3. Compile and train
# ---------------------------------------------------------------------------
model = create_model()

model.compile(
    optimizer=Nadam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    verbose=1,
)


# ---------------------------------------------------------------------------
# 4. Evaluate and display results
# ---------------------------------------------------------------------------
score = model.evaluate(X_test, y_test, verbose=0)

results = pd.DataFrame(
    {
        "Optimizer": ["Nadam"],
        "Loss": [score[0]],
        "Accuracy": [score[1]],
    }
)

print(results.to_string(index=False))
