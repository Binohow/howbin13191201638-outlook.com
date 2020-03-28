""" 熟悉数据集 """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import numpy as np
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data()
