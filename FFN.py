import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_prop(self, input):
        pass

    def back_prop(self, output_grad, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    def forward_prop(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def back_prop(self, output_grad, learning_rate):
        input_grad = np.dot(output_grad, self.weights.T)
        weights_grad = np.dot(self.input.T, output_grad)
        self.weights = self.weights - learning_rate * weights_grad
        self.bias = self.bias - learning_rate * output_grad
        return input_grad


class Activation(Layer):
    """Takes function and its derivitive"""
    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_prime = activation_der

    def forward_prop(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def back_prop(self, output_grad, learning_rate):
        return np.multiply(self.activation_prime(self.input), output_grad)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_der = lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        super().__init__(sigmoid, sigmoid_der)


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_der = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_der)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.max(x, 0)
        relu_der = lambda x: 1 if x.any() > 0 else 0
        super().__init__(relu, relu_der)


# Loss function (MSE):
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_der(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)