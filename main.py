import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_DIR = 'data/train'  # Make sure this path is correct!
TEST_DIR = 'data/test'
IMG_SIZE = (48, 48)       # FER2013 images are 48x48
BATCH_SIZE = 64
EPOCHS = 25               # You can increase this to 50 later if needed

print("Checking TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================================
# 2. DATA LOADING & AUGMENTATION
# ==========================================
# We use ImageDataGenerator to load images on the fly from the folders

# Training Data: Add random variety (augmentation) to make model smarter
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixels to 0-1
    rotation_range=15,       # Rotate slightly
    zoom_range=0.15,         # Zoom in/out
    width_shift_range=0.1,   # Shift left/right
    height_shift_range=0.1,  # Shift up/down
    horizontal_flip=True,    # Flip mirror images
    fill_mode='nearest'
)

# Test Data: Only rescale! Do not scramble the test data.
test_datagen = ImageDataGenerator(rescale=1./255)

print("\n--- Loading Training Data ---")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',  # Important: Dataset is B&W
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("\n--- Loading Test Data ---")
validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Don't shuffle so we can plot confusion matrix later
)

# ==========================================
# 3. BUILD THE CNN MODEL
# ==========================================
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 Emotions
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("\n--- Starting Training ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Save the model file
model.save('emotion_model.h5')
print("Model saved as emotion_model.h5")

# ==========================================
# 5. VISUALIZATION & EVALUATION
# ==========================================

# A. Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('accuracy_plot.png') # Saves the image for your report
plt.show()

# B. Confusion Matrix
print("\n--- Generating Confusion Matrix ---")
# Get true labels and predicted labels
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Create Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Print detailed report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))