'''
RNN Regressor 循环神经网络
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

batch_start = 0
time_steps = 20
batch_size =50
input_size = 1
output_size = 1
cell_size = 20
lr = 0.06

def get_batch():
    global batch_start, time_steps
    xs = np.arange(batch_start, batch_start+time_steps*batch_size).reshape((batch_size, time_steps))/(10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += time_steps
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

model = Sequential()
model.add(
    LSTM(
        batch_input_shape=(batch_size, time_steps, input_size),
        units=output_size,
        return_state=True,
        stateful=True
    )
)
model.add(TimeDistributed(Dense(output_size)))
adam=Adam(lr)
model.compile(optimizer=adam, loss='mse')

print('Training ------------')
for step in 501:
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch,Y_batch)
    pred = model.predict(X_batch, batch_size)
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:time_steps], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)