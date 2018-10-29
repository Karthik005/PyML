import data_util as du
import Perceptron as pcp
import matplotlib.pyplot as plt
import numpy as np


def show_class_plot():
    df = du.read_data_file("iris.data", """F:/Excelsior/PyML/code/data/iris/""" )
    y = df.iloc[0:100,4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()

def perceptron_driver():
    df = du.read_data_file("iris.data", """F:/Excelsior/PyML/code/data/iris/""" )
    y = df.iloc[0:100,4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    ppn = pcp.Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    # plt.scatter(x[:50, 0, x[]])



if __name__ == '__main__':
    perceptron_driver()