### Saliency Map for XTiny-FastKAN
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to compute saliency map
def compute_saliency_map(model, input_image, target_class_index=None):
    """
    Computes a basic gradient-based saliency map for the input image.

    Parameters:
    - model: Trained XTiny-FastKAN model
    - input_image: A single input image of shape (H, W, C)
    - target_class_index: Optional. If None, the predicted class is used.

    Returns:
    - saliency_map: 2D array of the same HxW size
    """
    input_image = tf.convert_to_tensor(input_image[None, ...])  # Add batch dimension
    input_image = tf.cast(input_image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predictions = model(input_image)
        class_index = target_class_index if target_class_index is not None else tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    # Compute gradients of loss with respect to input image
    gradients = tape.gradient(loss, input_image)

    # Take absolute value of gradients and max over channels
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]

    # Normalize between 0 and 1 for visualization
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)

    return saliency.numpy()

# Visualize the saliency map
def show_saliency(input_image, saliency_map, cmap="jet"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(input_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(input_image, alpha=0.6)
    axes[1].imshow(saliency_map, cmap=cmap, alpha=0.6)
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# Example: Visualize saliency for a test sample
index = 0  # Choose any test image
input_image = test_images[index]      # Shape: (32, 32, 3)
saliency_map = compute_saliency_map(model, input_image)
show_saliency(input_image, saliency_map)
