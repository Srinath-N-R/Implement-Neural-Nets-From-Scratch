import logging

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)-8s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(y, x) * 0.1 for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((y, 1)) for y in layers[1:]]
        logging.info("Initialized Neural Network with architecture: %s", "->".join(map(str, layers)))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def calculate_loss(self, y_pred, y_true):
        m = y_true.shape[1]
        loss = -np.sum(np.multiply(y_true, np.log(y_pred + 1e-9))) / m
        return loss

    def forward_propagation(self, x):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activation = self.sigmoid(z) if w is not self.weights[-1] else self.softmax(z)
            activations.append(activation)
        return activations, zs

    def backward_propagation(self, x, y, activations, zs):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta = activations[-1] - y
        nabla_b[-1], nabla_w[-1] = delta, np.dot(delta, activations[-2].T)
        for l in range(2, len(self.layers)):
            z, sp = zs[-l], self.sigmoid_derivative(zs[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l], nabla_w[-l] = delta, np.dot(delta, activations[-l-1].T)
        return nabla_w, nabla_b

    def update_parameters(self, nabla_w, nabla_b):
        self.weights = [w - self.learning_rate * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.learning_rate * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, input_data, targets, epochs):
        for epoch in range(epochs):
            losses = []
            for i in range(input_data.shape[1]):
                x, y = input_data[:, [i]], np.eye(self.layers[-1])[targets[i]].reshape(-1, 1)
                activations, zs = self.forward_propagation(x)
                losses.append(self.calculate_loss(activations[-1], y))
                nabla_w, nabla_b = self.backward_propagation(x, y, activations, zs)
                self.update_parameters(nabla_w, nabla_b)
            logging.info("Epoch %d/%d, Loss: %.4f", epoch + 1, epochs, np.mean(losses))

    def predict(self, input_data):
        results = []
        for x in input_data.T:
            activations, _ = self.forward_propagation(x.reshape(-1, 1))
            results.append(np.argmax(activations[-1]))
        return results


def train_network(nn, trainX_flat, trainY, epochs):
    """
    Trains the neural network.

    Args:
        nn: An instance of NeuralNetwork
        trainX_flat: Training data, flattened and normalized
        trainY: Training labels
        epochs: Number of epochs to train for
    """
    nn.train(trainX_flat, trainY, epochs)
    print("Training completed.")


def evaluate_network(nn, testX_flat, testY):
    """
    Evaluates the neural network on the test dataset.

    Args:
        nn: An instance of NeuralNetwork
        testX_flat: Test data, flattened and normalized
        testY: Test labels
    Returns:
        accuracy: The accuracy of the model on the test set
    """
    predictions = nn.predict(testX_flat)
    accuracy = np.mean(predictions == testY) * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy


def plot_example_predictions(nn, data, labels, num_examples=10):
    """
    Plots example predictions from the neural network.

    Args:
        nn: An instance of NeuralNetwork
        data: Data to predict on, flattened and normalized
        labels: Actual labels for the data
        num_examples: Number of examples to plot
    """
    fig, axes = plt.subplots(1, num_examples, figsize=(20, 2))
    predictions = nn.predict(data[:, :num_examples])
    for i in range(num_examples):
        ax = axes[i]
        ax.imshow(data[:, i].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {labels[i]}, Pred: {predictions[i]}')
        ax.axis('off')
    plt.show()


# This function is optional and can be used for more detailed configuration or experiments
def load_and_preprocess_data():
    """
    Loads and preprocesses the fashion MNIST dataset.

    Returns:
        trainX_flat, trainY, testX_flat, testY: Preprocessed training and test data and labels
    """
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    trainX_flat = trainX.reshape(trainX.shape[0], -1).T / 255.0
    testX_flat = testX.reshape(testX.shape[0], -1).T / 255.0
    return trainX_flat, trainY, testX_flat, testY
