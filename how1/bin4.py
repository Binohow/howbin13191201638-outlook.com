# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:24:10 2020

@author: Howbin
"""
# %%
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
# %%
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# %%
encoded = to_categorical(integer_encoded)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)