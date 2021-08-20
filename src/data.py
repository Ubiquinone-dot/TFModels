# Processes data from data/model2. The data is generated manually so there is a function to do so here
# all loading functions must return data in form of: (train_x, train_labels), (test_x, test_labels)
# process functions are void and should store data in the data directory
from src.dependencies import *


def load_fashion_MNIST(MODEL = 1):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_x, train_labels), (test_x, test_labels) = fashion_mnist.load_data()
    train_x = train_x / 255.
    test_x = test_x / 255.

    CONVNETS = (2, 3, 5)  # Which models use softmax
    if MODEL in CONVNETS:
        train_x = np.expand_dims(train_x, -1)
        test_x = np.expand_dims(test_x, -1)


        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)


    return (train_x, train_labels), (test_x, test_labels)


load_fashion_MNIST()
