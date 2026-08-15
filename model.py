"""
model.py — Digit Recognition CNN training script (enhanced)
==============================================================
Same model architecture as the original, with evaluation additions that
matter for a Data Analyst/Data Engineer portfolio: this isn't just
"trained a model," it's "trained a model AND properly evaluated it" —
confusion matrix, per-class precision/recall/F1, and training curves,
saved as PNGs you can drop straight into a README or resume portfolio
page.

NOTE: This requires `keras`, `tensorflow`, `tensorflowjs`, `matplotlib`,
and `scikit-learn`, plus internet access to download MNIST the first
time you run it. It was NOT re-executed in this build environment (no
internet access to download MNIST here) — the original model files you
uploaded are untouched and still used for the web app. Run this
yourself if you want fresh evaluation plots for your portfolio/README.

Run:
    pip install tensorflow tensorflowjs matplotlib scikit-learn
    python model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils as np_utils
import tensorflowjs as tfjs

""" Obtaining the data """
(x_train, y_train), (x_test, y_test) = mnist.load_data()

""" Preprocessing """
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")
x_train /= 255
x_test /= 255

y_test_labels = y_test.copy()  # keep original integer labels for the confusion matrix later
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

""" Creating the model """
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=10, activation="softmax"))
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

""" Train, keeping the history for the training-curve plot """
history = classifier.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=200, epochs=10,
)

""" Basic evaluation (same as original) """
scores = classifier.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: {:.2f}%".format(scores[1] * 100))
print("Error: {:.2f}%".format((1 - scores[1]) * 100))

# ---------------------------------------------------------------------
# EVALUATION ADDITIONS (new)
# ---------------------------------------------------------------------

# 1. Training curves: accuracy + loss over epochs
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss over epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("evaluation_training_curves.png", dpi=150)
print("Saved evaluation_training_curves.png")

# 2. Confusion matrix — shows exactly which digits get confused with which
#    (e.g. 4s misread as 9s, 7s misread as 1s are the classic MNIST mix-ups)
y_pred_probs = classifier.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test_labels, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Digit Classification")
plt.tight_layout()
plt.savefig("evaluation_confusion_matrix.png", dpi=150)
print("Saved evaluation_confusion_matrix.png")

# 3. Per-class precision/recall/F1 — more informative than one accuracy number,
#    shows whether any single digit is systematically harder for the model
report = classification_report(y_test_labels, y_pred, digits=3)
print("\nClassification report (per digit):\n")
print(report)
with open("evaluation_classification_report.txt", "w") as f:
    f.write(report)
print("Saved evaluation_classification_report.txt")

""" Save the model for the web app (unchanged from original) """
tfjs.converters.save_keras_model(classifier, "model")
print("Model exported to model/ for the web app")
