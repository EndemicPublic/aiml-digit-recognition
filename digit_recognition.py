"""
Handwritten Digit Recognition using MNIST Dataset
A beginner-friendly AIML project for B.Tech first-year students

Author: Your Name
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 50)
print("HANDWRITTEN DIGIT RECOGNITION")
print("=" * 50)

# Step 1: Load the MNIST dataset
print("\n[1] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"    Training samples: {len(x_train)}")
print(f"    Testing samples: {len(x_test)}")

# Step 2: Preprocess the data
print("\n[2] Preprocessing data...")
# Normalize pixel values to range 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images from 28x28 to 784 (flat vector)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
print("    Data normalized and reshaped!")

# Step 3: Build the neural network model
print("\n[3] Building neural network model...")
model = keras.Sequential([
    layers.Input(shape=(784,)),           # Input layer (28x28 = 784 pixels)
    layers.Dense(128, activation='relu'), # Hidden layer with 128 neurons
    layers.Dropout(0.2),                  # Prevents overfitting
    layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax') # Output layer (10 digits: 0-9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("    Model architecture:")
model.summary()

# Step 4: Train the model
print("\n[4] Training the model...")
print("    This may take 2-3 minutes...")
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# Step 5: Evaluate the model
print("\n[5] Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n    Test Accuracy: {test_acc * 100:.2f}%")
print(f"    Test Loss: {test_loss:.4f}")

# Step 6: Make predictions on sample images
print("\n[6] Sample Predictions:")
predictions = model.predict(x_test[:5], verbose=0)
for i in range(5):
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]
    status = "✓" if predicted_label == actual_label else "✗"
    print(f"    Image {i+1}: Predicted={predicted_label}, Actual={actual_label} {status}")

# Step 7: Visualize training history
print("\n[7] Generating plots...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.savefig('training_history.png', dpi=150)
print("    Plots saved as 'training_history.png'")

# Step 8: Display sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = np.argmax(predictions[i])
    ax.set_title(f'Pred: {pred}')
    ax.axis('off')
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
print("    Sample predictions saved as 'sample_predictions.png'")

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 50)
print("\nNext steps to try:")
print("1. Add more layers to improve accuracy")
print("2. Try different activation functions")
print("3. Save and load your trained model")
print("4. Test with your own handwritten digits")
