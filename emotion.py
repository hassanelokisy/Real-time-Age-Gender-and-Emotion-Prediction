import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import Sequential, layers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

TRAIN_PATH = 'data/emotion_detection_data/train'
VALIDATION_PATH = 'emotion_detection_data/test'
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
DROUPOUT = 0.1
EPOCHS = 50

train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
)

validation_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255
)

train_generator = train_data_gen.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

validation_generator = validation_data_gen.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

# ####### TO check if the images was loaded correctly

# imgs , labels  = train_generator.__next__()
#
# import random
# i = random.randint(0,BATCH_SIZE)
# img = imgs[i]
# label = class_labels[labels[i].argmax()]
# plt.imshow(img, cmap='gray')
# plt.title(label)
# plt.show()


# Creating the model
def my_model():
    model = Sequential()
    model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(Conv2D(32,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='valid',
                     activation='relu'),
              )
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(DROUPOUT))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROUPOUT))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(DROUPOUT))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model

model = my_model()
import tensorflow.keras.callbacks as callbackss


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

callbacks = [
    ModelCheckpoint('models/emotion_detection_model_50epochs.h5', monitor='val_loss', verbose=1, save_best_only=True,),
    EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
]


history = model.fit(train_generator,
                    epochs=10,
                    validation_data=validation_generator,
                    callbacks=callbacks
                    )

model.save('models/emotion_detection_model_50epochs.h5')