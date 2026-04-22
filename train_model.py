import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# 1. Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1  

print("--- Stage 1: Loading Data ---")
# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    horizontal_flip=True, 
    validation_split=0.2 
)

# This looks for images in dataset/train/real and dataset/train/fake
try:
    train_generator = train_datagen.flow_from_directory(
        'dataset/train', 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary', 
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'dataset/train', 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary', 
        subset='validation'
    )
except Exception as e:
    print(f"Error loading images: {e}")
    print("Make sure your images are in dataset/train/real and dataset/train/fake")
    exit()

print("\n--- Stage 2: Building Model ---")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Stage 3: Training ---")
print("This may take a few minutes. Please do not close the terminal...")
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save the final brain
model.save('deepfake_model.h5')
print("\n✅ SUCCESS! Your model is saved as deepfake_model.h5")