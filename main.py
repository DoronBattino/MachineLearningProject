import pandas as pd
import numpy as np
from tensorflow.python import tf2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from FFN import Dense, ReLU, Tanh, Sigmoid, mse, mse_der

# Data preprocess:
data = pd.read_csv('joined_dataframe.csv')
data = data.drop(['Num_Acc', 'Total_grav'], axis=1)
data = data.replace('[^\d.]', '', regex=True).astype(float)  # fix strings
scaler = MinMaxScaler()
X = scaler.fit_transform(data.drop(['num_of_inj'], axis=1))
y = data['num_of_inj'].apply(lambda x: x if x < 5 else 5)  # 10 classes of output (0-5, 5+)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 1, 17)
X_train = X_train.astype('float32')
y_train = np_utils.to_categorical(y_train)
X_test = X_test.reshape(X_test.shape[0], 1, 17)
X_test = X_test.astype('float32')
y_test = np_utils.to_categorical(y_test)

# Network structure:
network = [Dense(17, 10), Tanh(), Dense(10, 6), Tanh()]

iterations = 10
learning_rate = 0.1

# Train:
for iteration in range(iterations):
    error = 0
    for x, y in zip(X_train, y_train):
        layer_output = x
        for layer in network:
            layer_output = layer.forward_prop(layer_output)
        # Calculate error at last layer:
        error = error + mse(y, layer_output)
        # Backpropagation:
        gradient = mse_der(y, layer_output)
        for layer in reversed(network):
            gradient = layer.back_prop(gradient, learning_rate)
    error = error / len(X_train)
    print('%d/%d, error=%f' % (iteration + 1, iterations, error))

# Test:
good = 0
for x, y in zip(X_test, y_test):
    layer_output = x
    for layer in network:
        layer_output = layer.forward_prop(layer_output)
    if np.argmax(layer_output) == np.argmax(y):
        good = good + 1
    print('pred:', np.argmax(layer_output), '\ttrue:', np.argmax(y))
print('Total sucess is:')
print(good / len(y_test))