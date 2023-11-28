#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:45:13 2023

@author: erc_magnesia_raj
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import imageio.v3 as iio
from random import shuffle

Ntrain = 2000
train_images = np.zeros((2*Ntrain,374,500))
train_labels = []

#Load training data
for j in range(Ntrain):
    fdir = 'phase_maps/'
    fname = 'phase' + str(j+1) + '_train.png'
    #Class signal
    imsig = iio.imread(fdir+fname)
    imsig = imsig[:,:,1]
    train_images[j,:,:] = imsig
    train_labels.append([1])

    fdir = 'noise/'
    fname = 'noise' + str(j+1) + '_train.png'
    #Class noise
    imnoise = iio.imread(fdir+fname)
    imnoise = imnoise[:,:,1]
    train_images[j+1,:,:] = imnoise
    train_labels.append([0])

#Shuffle training set
train_labels = np.array(train_labels)
ind_list = [i for i in range(2*Ntrain)]
shuffle(ind_list)
train_images_new = train_images[ind_list, :,:]
train_labels_new = train_labels[ind_list,]

#Build a convolutional neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu',\
                        input_shape=(374, 500, 1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam',\
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
metrics=['accuracy'])

#Train model
history = model.fit(train_images, train_labels, epochs=10)
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
