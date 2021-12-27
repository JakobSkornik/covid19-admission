import numpy as np


class StochasticGradientDescent:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=1.0, decay=0.0):
        """Initialzes SGD class."""

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        """Executes before updating params."""

        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """Updates weights and biases."""

        layer.weights -= self.current_learning_rate * layer.d_weights
        layer.biases -= self.current_learning_rate * layer.d_biases

    def post_update_params(self):
        """Executes after updating params."""

        self.iterations += 1
