import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from keras_vggface.vggface import VGGFace

# Set TensorFlow to use the Metal GPU backend
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

# Define parameters
input_shape = (224, 224, 3)
num_classes = 2  # Change this to the number of classes in your dataset
batch_size = 32
epochs = 10

# Load the pre-trained VGGFace ResNet-50 model
base_model = VGGFace(model='resnet50', include_top=False, input_shape=input_shape)

# Create a new model and add the pre-trained model as the base
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))  # Customize the number of neurons in the fully connected layers
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '/Users/joaofulgencio/Documents/PUC/Projeto Integrador VI/PUC/projetointegradorvi/training/personA',  # Path to your training data
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'validation_data',  # Path to your validation data
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the trained model
model.save('facial_recognition_model.h5')
