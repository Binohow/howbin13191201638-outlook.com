""" Regressor 回归 """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

X = np.linspace(-1,1,200)
np.random.shuffle(X)
Y =0.5*X+2+np.random.normal(0,0.05,200)
# plot data
plt.scatter(X, Y)
plt.show()

X_train,Y_train = X[:160],Y[:160]
X_test,Y_test = X[160:],Y[160:]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 ==0:
        print('train cost: ', cost)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.show()
plt.scatter(X_test, Y_pred)
plt.show()
plt.scatter(X_test, Y_test)
plt.scatter(X_test, Y_pred)
plt.show()