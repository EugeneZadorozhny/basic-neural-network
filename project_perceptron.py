import numpy as np


class Perceptron:
    delta = 0
    activity_value = 0
    weight_delta = 0
    activation_value = 0

    def __init__(
        self,
        weights,
        bias,
        eta=1,
        activity_function=lambda x, y: x * y,
        activation_function=lambda x: 1.0 / (1 + np.exp(-1 * x)),
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.activity_function = activity_function
        self.activation_function = activation_function
        self.eta = eta
        self.previous_delta_weights = np.zeros(len(weights))

    def calc_activity(self, input):
        activity = 0
        for i, w in zip(input, self.weights):
            activity += self.activity_function(i, w)
        self.activity_value = activity + self.bias

    def calc_activation(self, input):
        self.calc_activity(input)
        self.activation_value = self.activation_function(self.activity_value)

    def update_weights_and_bias(self, desired, input, momentum):
        y = self.activation_value
        self.delta = (desired - y) * y * (1 - y)
        for i in range(len(self.weights)):
            self.weights[i] = (
                self.weights[i]
                + self.eta * self.delta * input[i]
                + momentum * self.previous_delta_weights[i]
            )
            self.previous_delta_weights[i] = self.eta * self.delta * input[i]
        self.bias = self.bias + self.eta * self.delta

    def update_hidden_weights_and_bias(self, desired, delta, input, momentum):
        y = self.activation_value
        self.delta = (desired * delta) * y * (1 - y)
        for i in range(len(self.weights)):
            self.weights[i] = (
                self.weights[i]
                + self.eta * self.delta * input[i]
                + momentum * self.previous_delta_weights[i]
            )
            self.previous_delta_weights[i] = self.eta * self.delta * input[i]
        self.bias = self.bias + self.eta * self.delta
