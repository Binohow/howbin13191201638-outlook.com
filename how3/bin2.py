# -*- coding: utf-8 -*-
""" Classifier 分类 """
from keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import RMSprop
# %% 感受数据
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

for num in range(50):
    plt.imshow(X_train[num,:,:],cmap='Greys_r')
    plt.axis('off')
    plt.show()
# %% 数据处理
X_train = X_train.reshape(X_train.shape[0],-1)/255
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
X_test = X_test.reshape(X_test.shape[0],-1)/255
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
# %% 建立模型
model = Sequential(
    [
        Dense(32, input_dim = 784 ),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ]
)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# %% 编译模型
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
# %% 训练模型
print('Training ------------')
model.fit(X_train,Y_train,epochs=2,batch_size=32)
# %% 模型评估
print('\nTesting ------------')
loss,accuracy = model.evaluate(X_test,Y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

X_predict = X_test[:10,:]
X_predict = X_predict.reshape(10,28,28)
plt.imshow(X_predict[0,:,:],cmap='Greys_r')
plt.axis('off')
plt.show()
img_0 = X_test[0,:].reshape(1,784)
output = model.predict(img_0)
result = np.argmax(output)
print(result)