import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import Sequential
import pathlib
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

directory = "data/age_gender_data/UTKFace/"

images = []
age = []
gender = []
for img in os.listdir(directory):
    ages = img.split("_")[0]
    genders = img.split("_")[1]
    img = cv2.imread(str(directory) + "/" + str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(np.array(img))
    age.append(np.array(ages))
    gender.append(np.array(genders))

age = np.array(age, dtype=np.int64)
images = np.array(images)  # Forgot to scale image for my training. Please divide by 255 to scale.
gender = np.array(gender, np.uint64)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)



age_model = Sequential()
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200, 200, 3)))
# age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPooling2D(pool_size=3, strides=2))

age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
# age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPooling2D(pool_size=3, strides=2))

age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
# age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPooling2D(pool_size=3, strides=2))

age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
# age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPooling2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))


age_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.mse,
                  metrics=['mae'])

age_callbacks = [
    callbacks.ModelCheckpoint('models/age_model_10epochs.h5', monitor='val_loss', verbose=1, save_best_only=True,),
    callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
]

histoty_age = age_model.fit(x_train_age, y_train_age,
                            validation_data=(x_test_age, y_test_age),
                            epochs=10,
                            callbacks=age_callbacks,
                            )


# gender model

gender_model = Sequential()

gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200,200,3)))

gender_model.add(MaxPooling2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPooling2D(pool_size=3, strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))


gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# save_model_callback_gender = callbacks.ModelCheckpoint('models/gender_model_50epochs.h5',
#                                                  save_weights_only=True,
#                                                  save_best_only=False,
#                       
#                                  )

gender_callbacks = [
    callbacks.ModelCheckpoint('models/gender_model_10epochs.h5', monitor='val_loss', verbose=1, save_best_only=True,),
    callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
]

history_gender = gender_model.fit(x_train_gender, y_train_gender,
                          validation_data=(x_test_gender, y_test_gender),
                          epochs=10,
                          callbacks=gender_callbacks
                          )






