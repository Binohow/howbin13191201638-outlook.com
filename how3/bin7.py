"""
情感分析实例：IMDB 影评情感分析
CNN
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten
from keras.models import Sequential

seed = 7
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs = 2

def create_model():
    model = Sequential()
    model.add(
        Embedding(top_words, output_dim=out_dimension, input_length=max_words)
    )
    model.add(
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model    

np.random.seed(seed=seed)
(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=top_words)
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_validation = sequence.pad_sequences(x_validation, maxlen=max_words)

# 生成模型
model = create_model()
model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=batch_size,epochs=epochs, verbose=2)

