import sys
from time import time_ns
import matplotlib.pyplot as plt
from runtime_metrics import RuntimeMetrics
from sklearn import metrics


# predict output of neural network and plot ROC curves based on different threshold values
def nn_roc_predict(
    threshold_start,
    threshold_end,
    increment,
    hidden_layer,
    output_layer,
    x,
    y,
    training_runtime_metric,
    title,
):
    specificity = []
    sensitivity = []
    x_axis = []
    testing_start_time = time_ns()
    iterations = int((threshold_end - threshold_start) / increment)
    for t in range(iterations):
        prediction = []
        tp = 0
        tn = 0
        p = 0
        n = 0
        threshold_start += increment
        x_axis.append(threshold_start)

        for i in range(len(x)):
            out = output_layer.get_layer_output(hidden_layer.get_layer_output(x[i]))
            if out[0] > threshold_start:
                prediction.append(1)
            else:
                prediction.append(0)
            if prediction[i] == y[i]:
                if prediction[i] == 1:
                    tp += 1
                else:
                    tn += 1
            if y[i] == 1:
                p += 1
            else:
                n += 1

        if p == 0:
            sensitivity.append(0)
        else:
            sensitivity.append(tp / p)
        if n == 0:
            specificity.append(0)
        else:
            specificity.append(tn / n)

    ##########################
    # Testing Network Runtime Metrics
    ##########################
    testing_end_time = time_ns()
    testing_data_size = sys.getsizeof(x) + sys.getsizeof(y)
    testing_runtime_metric = RuntimeMetrics(
        testing_data_size, testing_end_time - testing_start_time
    )

    ##########################
    # Program Runtime Metrics
    ##########################
    print(
        f"\nTraining Time: {training_runtime_metric.get_runtime() * pow(10, -6)} milliseconds"
    )
    print(f"Training Data Size: {training_runtime_metric.get_size()} bytes")
    print(
        f"Testing Time: {testing_runtime_metric.get_runtime() * pow(10, -6)} milliseconds"
    )
    print(f"Testing Data Size: {testing_runtime_metric.get_size()} bytes\n")

    # plot ROC curves
    plt.plot(x_axis, sensitivity, "g", label="sensitivity")
    plt.plot(x_axis, specificity, "r", label="specificity")
    plt.axvline(x=0.4)
    plt.ylabel("Rate")
    plt.xlabel("Threshold")
    plt.title(title + " ROC")
    plt.legend()
    plt.show()


# predict output of single perceptron and plot ROC curves based on different threshold values
def per_roc_predict(
    threshold_start,
    threshold_end,
    increment,
    perceptron,
    x,
    y,
    training_runtime_metric,
    title,
):
    specificity = []
    sensitivity = []
    x_axis = []
    testing_start_time = time_ns()
    iterations = int((threshold_end - threshold_start) / increment)
    for t in range(iterations):
        predict = []
        tp = 0
        tn = 0
        p = 0
        n = 0
        threshold_start += increment
        x_axis.append(threshold_start)

        for i in range(len(x)):
            perceptron.calc_activation(x[i])
            out = perceptron.activation_value
            if out > threshold_start:
                predict.append(1)
            else:
                predict.append(0)
            if predict[i] == y[i]:
                if predict[i] == 1:
                    tp += 1
                else:
                    tn += 1
            if y[i] == 1:
                p += 1
            else:
                n += 1

        if p == 0:
            sensitivity.append(0)
        else:
            sensitivity.append(tp / p)
        if n == 0:
            specificity.append(0)
        else:
            specificity.append(tn / n)

    ##########################
    # Testing Network Runtime Metrics
    ##########################
    testing_end_time = time_ns()
    testing_data_size = sys.getsizeof(x) + sys.getsizeof(y)
    testing_runtime_metric = RuntimeMetrics(
        testing_data_size, testing_end_time - testing_start_time
    )

    ##########################
    # Program Runtime Metrics
    ##########################
    print(
        f"\nTraining Time: {training_runtime_metric.get_runtime() * pow(10, -6)} milliseconds"
    )
    print(f"Training Data Size: {training_runtime_metric.get_size()} bytes")
    print(
        f"Testing Time: {testing_runtime_metric.get_runtime() * pow(10, -6)} milliseconds"
    )
    print(f"Testing Data Size: {testing_runtime_metric.get_size()} bytes\n")

    # plot ROC curves
    plt.plot(x_axis, sensitivity, "g", label="sensitivity")
    plt.plot(x_axis, specificity, "r", label="specificity")
    plt.axvline(x=0.4)
    plt.ylabel("Rate")
    plt.xlabel("Threshold")
    plt.title(title + " ROC")
    plt.legend()
    plt.show()


def per_confusion_matrix(x, y, perceptron, threshold):
    specificity = []
    sensitivity = []
    predict = []
    tp = 0
    tn = 0
    p = 0
    n = 0
    mse = []
    output = []
    for i in range(len(x)):
        perceptron.calc_activation(x[i])
        out = perceptron.activation_value
        output.append(round(float(out.copy()), 5))
        mse.append((y[i] - out) ** 2)
        if out > threshold:
            predict.append(1)
        else:
            predict.append(0)
        if predict[i] == y[i]:
            if predict[i] == 1:
                tp += 1
            else:
                tn += 1
        if y[i] == 1:
            p += 1
        else:
            n += 1

    if p == 0:
        sensitivity.append(0)
    else:
        sensitivity.append(tp / p)
    if n == 0:
        specificity.append(0)
    else:
        specificity.append(tn / n)
    ##########################
    # Confusion Matrix Plot
    ##########################
    confusion_mtrx = metrics.confusion_matrix(y, predict)
    tn, fp, fn, tp = confusion_mtrx.ravel()
    print(f"predicted output: {output}")
    print(f"threshold prediction: {predict}")
    print(f"actual:               {y}")
    print(f"mse: {sum(mse)/len(mse)}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_mtrx, display_labels=[0, 1]
    )
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()


def nn_confusion_matrix(x, y, hidden_layer, output_layer, threshold):
    specificity = []
    sensitivity = []
    predict = []
    tp = 0
    tn = 0
    p = 0
    n = 0
    mse = []
    output = []
    for i in range(len(x)):
        out = output_layer.get_layer_output(hidden_layer.get_layer_output(x[i]))
        output.append(round(float(out.copy()), 5))
        mse.append((y[i] - out) ** 2)
        if out > threshold:
            predict.append(1)
        else:
            predict.append(0)
        if predict[i] == y[i]:
            if predict[i] == 1:
                tp += 1
            else:
                tn += 1
        if y[i] == 1:
            p += 1
        else:
            n += 1

    if p == 0:
        sensitivity.append(0)
    else:
        sensitivity.append(tp / p)
    if n == 0:
        specificity.append(0)
    else:
        specificity.append(tn / n)
    ##########################
    # Confusion Matrix Plot
    ##########################
    confusion_mtrx = metrics.confusion_matrix(y, predict)
    tn, fp, fn, tp = confusion_mtrx.ravel()
    print(f"predicted output: {output}")
    print(f"threshold prediction: {predict}")
    print(f"actual:               {y}")
    print(f"mse: {sum(mse)/len(mse)}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_mtrx, display_labels=[0, 1]
    )
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()
