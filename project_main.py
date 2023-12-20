import sys
import project_roc_curves as roc
from time import time_ns
import project_layer as layer
import project_perceptron as per
from runtime_metrics import RuntimeMetrics

# Training Data
train_x = [
    [0.90, 0.87],
    [1.31, 0.75],
    [2.48, 1.14],
    [0.41, 1.87],
    [2.45, 0.52],
    [2.54, 2.97],
    [0.07, 0.09],
    [1.32, 1.96],
    [0.94, 0.34],
    [1.75, 2.21],
]
train_y = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]

# Testing Data
test_x = [
    [1.81, 1.02],
    [2.36, 1.60],
    [2.17, 2.08],
    [2.85, 2.91],
    [1.05, 1.93],
    [2.32, 1.73],
    [1.86, 1.31],
    [1.45, 2.19],
    [0.28, 0.71],
    [2.49, 1.52],
]
test_y = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0]


##########################
# Neural Network Method
##########################

# initial weights
hidden_weights = [[0.01, 0.005], [0.005, 0.01]]
output_weights = [[0.005, 0.01]]

# number of epochs
epoch = 300
eta = 1
momentum = 0

# initialize neural network
hidden1 = layer.Layer(False, hidden_weights, 0.01, eta)
output = layer.Layer(True, output_weights, 0.01, eta)

print("Neural Network")
print("Summary")
print(f"Epochs: {epoch}")
print(f"Eta: {eta}")
print(f"Momentum: {momentum}")
print(
    f"initial hidden weights :\n{hidden1.get_weights()} \nbias1: {hidden1.layer[0].bias} bias2: {hidden1.layer[1].bias}"
)
print(
    f"initial output weights :\n{output.get_weights()} \nbias: {output.layer[0].bias}"
)

# train neural network
training_start_time = time_ns()
for i in range(epoch):
    for index, sample in enumerate(train_x):
        next_input = hidden1.get_layer_output(sample)
        output_value = output.get_layer_output(next_input)
        old_output_weights = output.get_weights()[0]
        output.update_layer(train_y[index], next_input, momentum)
        hidden1.update_layer(output.get_delta()[0], old_output_weights, momentum)

print(
    f"final hidden weights :\n{hidden1.get_weights()} \nbias1: {hidden1.layer[0].bias} bias2: {hidden1.layer[1].bias}"
)
print(f"final output weights :\n{output.get_weights()} \nbias: {output.layer[0].bias}")

# Training Runtime Metrics
training_end_time = time_ns()
training_data_size = sys.getsizeof(train_x) + sys.getsizeof(train_y)
training_runtime_metric = RuntimeMetrics(
    training_data_size, training_end_time - training_start_time
)


# get prediction and ROC curves
threshold_start = 0
threshold_end = 1
increment = 0.001

roc.nn_roc_predict(
    threshold_start,
    threshold_end,
    increment,
    hidden1,
    output,
    train_x,
    train_y,
    training_runtime_metric,
    "NN Train Set",
)
roc.nn_confusion_matrix(train_x, train_y, hidden1, output, 0.48)

roc.nn_roc_predict(
    threshold_start,
    threshold_end,
    increment,
    hidden1,
    output,
    test_x,
    test_y,
    training_runtime_metric,
    "NN Test Set",
)
roc.nn_confusion_matrix(test_x, test_y, hidden1, output, 0.48)

##########################
# Single Perceptron Method
##########################

# initialize perceptron
perceptron = per.Perceptron([0.01, 0.01], 0.01, eta)

print("Single Perceptron")
print("Summary")
print(f"Epochs: {epoch}")
print(f"Eta: {eta}")
print(f"Momentum: {momentum}")
print(f"initial weights : {perceptron.weights} bias: {perceptron.bias}")

# train perceptron
training_start_time = time_ns()
for i in range(epoch):
    for index, sample in enumerate(train_x):
        perceptron.calc_activation(sample)
        perceptron.update_weights_and_bias(train_y[index], sample, momentum)

print(f"final weights : {perceptron.weights} bias: {perceptron.bias}")

# Training Runtime Metrics
training_end_time = time_ns()
training_data_size = sys.getsizeof(train_x) + sys.getsizeof(train_y)
training_runtime_metric = RuntimeMetrics(
    training_data_size, training_end_time - training_start_time
)


# get prediction and ROC curves
threshold_start = 0
threshold_end = 1
increment = 0.01

roc.per_roc_predict(
    threshold_start,
    threshold_end,
    increment,
    perceptron,
    train_x,
    train_y,
    training_runtime_metric,
    "Per Train Set",
)
roc.per_confusion_matrix(train_x, train_y, perceptron, 0.4)

roc.per_roc_predict(
    threshold_start,
    threshold_end,
    increment,
    perceptron,
    test_x,
    test_y,
    training_runtime_metric,
    "Per Test Set",
)
roc.per_confusion_matrix(test_x, test_y, perceptron, 0.4)
