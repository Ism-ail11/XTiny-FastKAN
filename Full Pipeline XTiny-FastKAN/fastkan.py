### Define our model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import matplotlib.pyplot as plt

# Define FastKAN model with modified architecture
class RadialBasisFunction(layers.Layer):
    def __init__(self, grid_min, grid_max, num_grids, **kwargs):
        super(RadialBasisFunction, self).__init__(**kwargs)
        self.grid = tf.cast(tf.linspace(grid_min, grid_max, num_grids), dtype=tf.float32)
        self.denominator = tf.cast((grid_max - grid_min) / num_grids, dtype=tf.float32)

    def call(self, x):
        return tf.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, grid_min, grid_max, num_grids, use_base_update, base_activation):
        super(FastKANLayer, self).__init__()
        self.norm = layers.LayerNormalization(axis=-1)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = layers.Dense(output_dim)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = layers.Dense(output_dim)

    def call(self, x):
        x_norm = self.norm(x)
        spline_basis = self.rbf(x_norm)
        spline_basis_flat = tf.reshape(spline_basis, [tf.shape(spline_basis)[0], -1])
        ret = self.spline_linear(spline_basis_flat)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class FastKAN(tf.keras.Model):
    def __init__(self, layers_hidden, grid_min=-1, grid_max=1, num_grids=2, use_base_update=False, base_activation=tf.nn.silu):
        super(FastKAN, self).__init__()
        self.layers_list = []
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers_list.append(FastKANLayer(in_dim, out_dim, grid_min, grid_max, num_grids, use_base_update, base_activation))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

# Define optimized KAN model
def create_kan():
    return FastKAN([32 * 32, 5, 2], grid_min=-1, grid_max=1, num_grids=2, use_base_update=True, base_activation=tf.nn.silu)

model = create_kan()

# Define optimizer and loss function
optimizer = optimizers.Adam(learning_rate=1e-3)
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Model Training Parameters
EPOCHS = 20
BATCH_SIZE = 256

# Define function to compute validation loss
def compute_loss(images, labels):
    logits = model(images)
    return loss_fn(labels, logits).numpy()

# Function to compute accuracy
def compute_accuracy(images, labels):
    logits = model(images)
    predictions = tf.argmax(logits, axis=1)
    correct = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
    return tf.reduce_mean(correct)

# Training Loop
train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")

    epoch_train_loss = 0
    num_batches = 0

    # Training loop
    for step in range(0, len(train_images), BATCH_SIZE):
        x_batch_train = train_images[step:step + BATCH_SIZE]
        y_batch_train = train_labels[step:step + BATCH_SIZE]

        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch_train_loss += loss_value.numpy()
        num_batches += 1

    # Compute average training loss
    avg_train_loss = epoch_train_loss / num_batches
    train_loss_list.append(avg_train_loss)

    # Compute accuracy
    train_acc = compute_accuracy(train_images, train_labels).numpy()
    val_acc = compute_accuracy(valid_images, valid_labels).numpy()

    # Compute validation loss
    val_loss = compute_loss(valid_images, valid_labels)
    val_loss_list.append(val_loss)

    print(f"✅ Training Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

model.save("kan32_32.keras")

# Plot Accuracy Curve
plt.figure(figsize=(10, 4))
plt.plot(range(1, EPOCHS+1), train_acc_list, label="Training Accuracy", marker='o')
plt.plot(range(1, EPOCHS+1), val_acc_list, label="Validation Accuracy", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")
plt.grid()
plt.show()

# Plot Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(range(1, EPOCHS+1), train_loss_list, label="Training Loss", marker='o')
plt.plot(range(1, EPOCHS+1), val_loss_list, label="Validation Loss", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.grid()
plt.show()

# Print the number of trainable and non-trainable parameters
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_variables])
non_trainable_params = total_params - trainable_params

print(f"📌 Total Parameters: {total_params}")
print(f"🔧 Trainable Parameters: {trainable_params}")
print(f"🚫 Non-Trainable Parameters: {non_trainable_params}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Replace with your true test labels and predicted labels
# test_labels: 1D array of true class indices (length = N)
# test_pred_labels: 1D array of predicted class indices (length = N)

# Define Tifinagh character labels in the correct order (index 0 to 32)
tifinagh_labels = [
    "ⴰ", "ⴱ", "ⴳ", "ⴷ", "ⴻ", "ⴼ", "ⴽ", "ⵀ", "ⵃ", "ⵄ", "ⵅ",
    "ⵇ", "ⵉ", "ⵊ", "ⵍ", "ⵎ", "ⵏ", "ⵓ", "ⵔ", "ⵕ", "ⵖ", "ⵙ",
    "ⵚ", "ⵛ", "ⵜ", "ⵟ", "ⵡ", "ⵢ", "ⵣ", "ⵤ", "ⵥ", "ⵯ", "ⵠ"
]

# Compute confusion matrix
cm = confusion_matrix(test_labels, test_pred_labels)

# Plot confusion matrix with Tifinagh labels
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=tifinagh_labels, yticklabels=tifinagh_labels)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

