# -*- coding: utf-8 -*-
""" Classifier 分类 """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend
backend.set_image_data_format('channels_first')

# %% 数据预处理
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 1, 28 , 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
Y_train =  np_utils.to_categorical(Y_train, num_classes=10)
Y_test =  np_utils.to_categorical(Y_test, num_classes=10)

# %% 建立模型
model = Sequential()

model.add(
    Convolution2D(
        batch_input_shape=(None, 1, 28, 28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same'
    )
)
model.add(Activation('relu'))

model.add(
    MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
    )
)
model.add(Activation('relu'))

model.add(Convolution2D(64, 5, strides=1, padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(2,2,'same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))
adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# %% 训练
print('Training ------------')
model.fit(X_train,Y_train,batch_size=64)

print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, Y_test)

# %% 评估
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)