from keras.datasets import fashion_mnist
import numpy as np
import torch
import FashNet
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score

# Constants
WEIGHT_PATH = "weights.pt"
IN_D = 28 * 28
OUT_D = 10

# Read in test data
(_, _), (X_test, y_test_np) = fashion_mnist.load_data()  ## Your code here
print(f"Input: {X_test.shape}")

# How many samples are we testing on?
n = X_test.shape[0]  ## Your code here

# Make each image into a long vector with range 0 to 1.0
X_test = X_test.reshape((-1, IN_D)).astype(np.float32) / 255.0  ## Your code here

# Convert to pytorch tensors
## Your code here
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(np.copy(y_test_np))


# Create the model
model = FashNet.FashNet(IN_D, OUT_D)  ## Your code here

# Read in the weights created in fashion_train.py
model.load_state_dict(torch.load("weights.pt"))

# We are doing inference (no need for backpropagation)
# Put the model in eval mode
## Your code here
model.eval()

# Make For each image, make a vector of 10
# "probabilities" -- one for each class
y_pred = model(X_test)  ## Your code here

classes = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
# Convert back to numpy so we can do stats on it
y_pred = y_pred.detach().numpy()
# Take the most likely for each image
predictions = np.argmax(y_pred, axis=1)  ## Your code here


# Get the accuracy
accuracy = accuracy_score(y_test, predictions)  ## Your code here
print(f"Accuracy on test data: {accuracy * 100.0:>2.2f}%")


# Make a confusion matrix
cm = confusion_matrix(y_test, predictions)  ## Your code here
print(f"Confusion: \n{cm}")

# Make it into a pretty plot
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_title("Fashion Confusion Matrix")
## Your code here
labels = classes
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
fig.savefig("confusion.png")
print("Wrote confusion.png")
