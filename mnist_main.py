## "The Hello World" of classification tasks: MNIST.

from sklearn.datasets import fetch_openml
mnist = fetch_openml('MNIST_784')
mnist = mnist['data']