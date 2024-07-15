import os
import re
import cv2 as cv
import urllib.request
import numpy as np
import time
import shutil
import zipfile
import matplotlib.pyplot as plt

from PIL import Image
from os import listdir
from os.path import isfile, join
from random import randrange
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

training_dir = r'train_dir'
test_dir = r'test_dir'

# Initiate data processing tools
training_dp = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10,
    shear_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1
)

test_dp = ImageDataGenerator(rescale=1./255)

training_data = training_dp.flow_from_directory(
    training_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_dp.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Parameters
num_conv_layers = 6
num_dense_layers = 3
layer_size = 64
num_training_epochs = 200
MODEL_NAME = 'soil'

model = Sequential()

# Initial Conv2D layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional convolution layers
for _ in range(num_conv_layers-1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

# Add output layer
model.add(Dense(11))  # Update the number of output neurons
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(
    training_data,
    epochs=num_training_epochs,
    validation_data=test_data,
    callbacks=[early_stopping]
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Check the number of epochs the model was trained for
print(f"Training stopped at epoch: {len(history.history['loss'])}")
