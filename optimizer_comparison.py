"""
MNIST Optimizer Comparison
--------------------------
Trains the same fully connected neural network on MNIST with five different
optimizers (SGD, Adam, RMSprop, Adagrad, Nadam) and reports which one gives
the best test accuracy.
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam


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

NUM_CLASSES = len(set(y_train))  # 10 digits


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

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    return model


# ---------------------------------------------------------------------------
# 3. Optimizers to compare
# ---------------------------------------------------------------------------
optimizers = {
    "SGD": SGD(),
    "Adam": Adam(),
    "RMSprop": RMSprop(),
    "Adagrad": Adagrad(),
    "Nadam": Nadam(),
}

optimizer_performance = {}


# ---------------------------------------------------------------------------
# 4. Train and evaluate each optimizer
# ---------------------------------------------------------------------------
for name, optimizer in optimizers.items():
    model = create_model()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        verbose=0,
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    optimizer_performance[name] = score

    print(f"Optimizer: {name}, Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")


# ---------------------------------------------------------------------------
# 5. Best optimizer (by test accuracy)
# ---------------------------------------------------------------------------
best_optimizer = max(optimizer_performance, key=lambda k: optimizer_performance[k][1])

print("\nBest Optimizer:", best_optimizer)
print("Best Score (loss, accuracy):", optimizer_performance[best_optimizer])
