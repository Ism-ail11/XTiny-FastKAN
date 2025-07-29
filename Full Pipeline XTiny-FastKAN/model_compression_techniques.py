### Model Compression Techniques

Dynamic Range Quantization
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import matplotlib.pyplot as plt

# Define representative dataset function for integer quantization
def representative_dataset():
    for i in range(100):  # Using a subset of training data
        yield [train_images[i: i + 1]]

# Ensure all variables are frozen before conversion
tf.lite.TFLiteConverter.experimental_new_converter = True

# Export the model to SavedModel format with a signature function
def serving_fn(inputs):
    return {"outputs": model(inputs)}

signature = tf.function(serving_fn).get_concrete_function(
    tf.TensorSpec(shape=[None, 32 * 32], dtype=tf.float32)
)

# Save model in Keras format instead of SavedModel
model.save("kan32_32.keras")

# Convert the model to TFLite format with different quantization methods
# 1. Default Quantization (Dynamic Range)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
dynamic_tflite_model = converter.convert()
with open("kan32_32_dynamic.tflite", "wb") as f:
    f.write(dynamic_tflite_model)
print("✅ Dynamic range quantized model saved as kan32_32_dynamic.tflite")

# 2. Float16 Quantization
converter.target_spec.supported_types = [tf.float16]
float16_tflite_model = converter.convert()
with open("kan32_32_float16.tflite", "wb") as f:
    f.write(float16_tflite_model)
print("✅ Float16 quantized model saved as kan32_32_float16.tflite")

# 3. Full Integer Quantization (with representative dataset)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
integer_tflite_model = converter.convert()
with open("kan32_32_integer.tflite", "wb") as f:
    f.write(integer_tflite_model)
print("✅ Full integer quantized model saved as kan32_32_integer.tflite")

# 4. Weight Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
weight_tflite_model = converter.convert()
with open("kan32_32_weight.tflite", "wb") as f:
    f.write(weight_tflite_model)
print("✅ Weight quantized model saved as kan32_32_weight.tflite")

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import matplotlib.pyplot as plt
from PIL import Image
import time  # Import time module for inference measurement

# Define representative dataset function for integer quantization
def representative_dataset():
    for i in range(100):  # Using a subset of training data
        yield [train_images[i: i + 1]]

# Ensure all variables are frozen before conversion
tf.lite.TFLiteConverter.experimental_new_converter = True

# Export the model to SavedModel format with a signature function
def serving_fn(inputs):
    return {"outputs": model(inputs)}

signature = tf.function(serving_fn).get_concrete_function(
    tf.TensorSpec(shape=[None, 32 * 32], dtype=tf.float32)
)

# Load the quantized integer model
interpreter = tf.lite.Interpreter(model_path="kan32_32_integer.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = Image.open("fig").convert("L")
image = image.resize((32, 32))
image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
image_array = image_array.reshape(1, 32 * 32)

# Measure inference time for a single image
start_time = time.time()

# Run inference
interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.int8))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)

end_time = time.time()
inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

# Define label names
label_names = { "ⴰ", "ⴱ", "ⴳ", "ⴷ", "ⴻ", "ⴼ", "ⴽ", "ⵀ", "ⵃ", "ⵄ", "ⵅ",
    "ⵇ", "ⵉ", "ⵊ", "ⵍ", "ⵎ", "ⵏ", "ⵓ", "ⵔ", "ⵕ", "ⵖ", "ⵙ",
    "ⵚ", "ⵛ", "ⵜ", "ⵟ", "ⵡ", "ⵢ", "ⵣ", "ⵤ", "ⵥ", "ⵯ", "ⵠ"}
predicted_label_name = label_names.get(predicted_label, "Unknown")

# Display the image with the predicted label
plt.figure(figsize=(4, 4))
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis("off")
plt.show()

# Print results
print(f"🖼️ Predicted Label: {predicted_label} ({predicted_label_name})")
print(f"⏱️ Inference Time: {inference_time:.2f} ms")

"""Full Integer Quantization"""

import time
import numpy as np
import tensorflow as tf

# Load the quantized dynamic model
interpreter = tf.lite.Interpreter(model_path="kan32_32_integer.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load test dataset
# Assuming test_images and test_labels are already loaded as numpy arrays
correct_predictions = 0
total_start_time = time.time()  # Start measuring total inference time

for i in range(len(test_images)):
    image_array = test_images[i].reshape(1, 32 * 32).astype(np.int8)
    label = test_labels[i]

    # Measure inference time for a single image
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    end_time = time.time()
    if predicted_label == label:
        correct_predictions += 1

total_end_time = time.time()
total_inference_time = (total_end_time - total_start_time) * 1000  # Convert to milliseconds
avg_inference_time = total_inference_time / len(test_images)  # Compute average per image

# Compute accuracy
accuracy = correct_predictions / len(test_images) * 100

# Print results
print(f"✅ Test Accuracy: {accuracy:.2f}%")
print(f"⏱️ Total Inference Time for {len(test_images)} Images: {total_inference_time:.2f} ms")
print(f"⏱️ Average Inference Time per Sample: {avg_inference_time:.2f} ms")

# Load the quantized integer model
interpreter = tf.lite.Interpreter(model_path="kan32_32_float16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = Image.open("fig.jpg").convert("L")
image = image.resize((32, 32))
image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
image_array = image_array.reshape(1, 32 * 32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)

# Define label names
label_names = { "ⴰ", "ⴱ", "ⴳ", "ⴷ", "ⴻ", "ⴼ", "ⴽ", "ⵀ", "ⵃ", "ⵄ", "ⵅ",
    "ⵇ", "ⵉ", "ⵊ", "ⵍ", "ⵎ", "ⵏ", "ⵓ", "ⵔ", "ⵕ", "ⵖ", "ⵙ",
    "ⵚ", "ⵛ", "ⵜ", "ⵟ", "ⵡ", "ⵢ", "ⵣ", "ⵤ", "ⵥ", "ⵯ", "ⵠ"}
predicted_label_name = label_names.get(predicted_label, "Unknown")

# Display the image with the predicted label
plt.figure(figsize=(4, 4))
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis("off")
plt.show()

print(f"🖼️ Predicted Label: {predicted_label_name}")

"""Float-16 Quantization"""

import time
import numpy as np
import tensorflow as tf

# Load the quantized Float16 model
interpreter = tf.lite.Interpreter(model_path="kan32_32_float16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load test dataset
# Assuming test_images and test_labels are already loaded as numpy arrays
correct_predictions = 0
total_start_time = time.time()  # Start measuring total inference time

for i in range(len(test_images)):
    image_array = test_images[i].reshape(1, 32 * 32).astype(np.float32)
    label = test_labels[i]

    # Measure inference time for a single image
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    end_time = time.time()

    if predicted_label == label:
        correct_predictions += 1

total_end_time = time.time()
total_inference_time = (total_end_time - total_start_time) * 1000  # Convert to milliseconds
avg_inference_time = total_inference_time / len(test_images)  # Compute average per image

# Compute accuracy
accuracy = correct_predictions / len(test_images) * 100

# Print results
print(f"✅ Test Accuracy: {accuracy:.2f}%")
print(f"⏱️ Total Inference Time for {len(test_images)} Images: {total_inference_time:.2f} ms")
print(f"⏱️ Average Inference Time per Sample: {avg_inference_time:.2f} ms")

# Load the quantized integer model
interpreter = tf.lite.Interpreter(model_path="kan32_32_weight.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = Image.open("fig.jpg").convert("L")
image = image.resize((32, 32))
image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
image_array = image_array.reshape(1, 32 * 32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)

# Define label names
label_names = {"ⴰ", "ⴱ", "ⴳ", "ⴷ", "ⴻ", "ⴼ", "ⴽ", "ⵀ", "ⵃ", "ⵄ", "ⵅ",
    "ⵇ", "ⵉ", "ⵊ", "ⵍ", "ⵎ", "ⵏ", "ⵓ", "ⵔ", "ⵕ", "ⵖ", "ⵙ",
    "ⵚ", "ⵛ", "ⵜ", "ⵟ", "ⵡ", "ⵢ", "ⵣ", "ⵤ", "ⵥ", "ⵯ", "ⵠ"}
predicted_label_name = label_names.get(predicted_label, "Unknown")

# Display the image with the predicted label
plt.figure(figsize=(4, 4))
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis("off")
plt.show()

print(f"🖼️ Predicted Label: {predicted_label_name}")

"""Weight-Only Quantization"""

import time
import numpy as np
import tensorflow as tf

# Load the quantized Weight-Only model
interpreter = tf.lite.Interpreter(model_path="kan32_32_weight.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load test dataset
# Assuming test_images and test_labels are already loaded as numpy arrays
correct_predictions = 0
total_start_time = time.time()  # Start measuring total inference time

for i in range(len(test_images)):
    image_array = test_images[i].reshape(1, 32 * 32).astype(np.float32)
    label = test_labels[i]

    # Measure inference time for a single image
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    end_time = time.time()

    if predicted_label == label:
        correct_predictions += 1

total_end_time = time.time()
total_inference_time = (total_end_time - total_start_time) * 1000  # Convert to milliseconds
avg_inference_time = total_inference_time / len(test_images)  # Compute average per image

# Compute accuracy
accuracy = correct_predictions / len(test_images) * 100

# Print results
print(f"✅ Test Accuracy: {accuracy:.2f}%")
print(f"⏱️ Total Inference Time for {len(test_images)} Images: {total_inference_time:.2f} ms")
print(f"⏱️ Average Inference Time per Sample: {avg_inference_time:.2f} ms")

# Load the quantized integer model
interpreter = tf.lite.Interpreter(model_path="kan32_32_dynamic.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = Image.open("/content/drive/MyDrive/UTARLDD_gray/UTARLDD_frames_gris/awake/0_11_frame_10201.jpg").convert("L")  # Convert to grayscale
image = image.resize((32, 32))
image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
image_array = image_array.reshape(1, 32 * 32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)

# Define label names
label_names = { "ⴰ", "ⴱ", "ⴳ", "ⴷ", "ⴻ", "ⴼ", "ⴽ", "ⵀ", "ⵃ", "ⵄ", "ⵅ",
    "ⵇ", "ⵉ", "ⵊ", "ⵍ", "ⵎ", "ⵏ", "ⵓ", "ⵔ", "ⵕ", "ⵖ", "ⵙ",
    "ⵚ", "ⵛ", "ⵜ", "ⵟ", "ⵡ", "ⵢ", "ⵣ", "ⵤ", "ⵥ", "ⵯ", "ⵠ"}
predicted_label_name = label_names.get(predicted_label, "Unknown")

# Display the image with the predicted label
plt.figure(figsize=(4, 4))
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Label: {predicted_label_name}")
plt.axis("off")
plt.show()

print(f"🖼️ Predicted Label: {predicted_label_name}")
