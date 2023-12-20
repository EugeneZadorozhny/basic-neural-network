import numpy as np
import project_perceptron as per


class Layer:
    def __init__(
        self,
        is_output,
        weights,
        bias,
        eta=1,
        activity_function=lambda x, y: x * y,
        activation_function=lambda x: 1.0 / (1 + np.exp(-1 * x)),
    ) -> None:
        self.layer = []
        self.output_flag = is_output
        self.output = np.zeros(len(weights))
        if self.output_flag:
            self.little_e = np.zeros(len(weights))
        for i in range(len(weights)):
            self.layer.append(
                per.Perceptron(
                    weights[i],
                    bias,
                    eta,
                    activity_function,
                    activation_function,
                )
            )

    def get_err_vector(self, desired):
        if self.output_flag:
            self.little_e = desired - self.output
        return self.little_e

    def get_layer_output(self, input):
        self.input = input
        for i in range(len(self.layer)):
            self.layer[i].calc_activation(input=input)
            self.output[i] = self.layer[i].activation_value
        return self.output

    def update_layer(self, desired, input, momentum):
        for i in range(len(self.layer)):
            if self.output_flag:
                self.layer[i].update_weights_and_bias(
                    desired=desired, input=input, momentum=momentum
                )
            else:
                self.layer[i].update_hidden_weights_and_bias(
                    desired=desired, delta=input[i], input=self.input, momentum=momentum
                )

    def get_weights(self):
        temp = []
        for i in range(len(self.layer)):
            temp.append(self.layer[i].weights)
        return np.copy(temp)

    def get_delta(self):
        temp = []
        for i in range(len(self.layer)):
            temp.append(self.layer[i].delta)
        return np.copy(temp)

    def get_big_e(self):
        return 0.5 * (self.little_e**2)
