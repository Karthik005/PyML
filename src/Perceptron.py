import numpy as np
import matplotlib as mtp


class Perceptron(object):
    """Perceptron classifier

    Parameters
    ----------------
    eta: float
        Learning rate (0.0 - 1.0)

    n_iter: int
        number of iterations

    Attributes
    ----------------
    w_: ld-array
        weights of model

    errors_: list
        num of misclassifications in every epoch

    """

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        """Fit training data

        Params
        --------------
        X: {array-like}, shape = [n_samples, n_features]
            training vectors

        Y: array like, shape = [n_samples]
            target values

        Returns
        ---------------
        self: object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                print(self.w_)
            self.errors_.append(errors)
        
        return self
    
    def net_input(self, X):
        """ Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after classification"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
