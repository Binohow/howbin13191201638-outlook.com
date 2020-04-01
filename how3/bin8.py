'''
RNN Classifier 循环神经网络
'''
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import  np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import  Adam

time_step = 28
input_size = 28
batch_size = 50
output_size = 10
cell_size = 50
LR = 0.001

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(
    SimpleRNN(
        batch_input_shape=(None, time_step, input_size),
        units=cell_size
    )
)

model.add(
    Dense(output_size)
)

model.add(Activation('softmax'))

adam = Adam(LR)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']    
)
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, epochs=2, verbose=2, validation_data=(X_test, y_test))
