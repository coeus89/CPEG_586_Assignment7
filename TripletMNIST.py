import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras
# import tensorflow.compat.v1 as tf
import matplotlib
import matplotlib.pyplot as plt
from Triplet import Triplet
import numpy as np
import sys


# tf.compat.v1.disable_v2_behavior()

def visualize(embed, labels):
    labelset = set(labels.tolist())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices, 0], embed[indices, 1], label=label, s=20)
    ax.legend()
    # fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    # __________example 1 siamese_____________________
    # Load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    mnist_test_labels = mnist.test.labels

    mnist2 = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist2.load_data()

    triplet = Triplet()
    triplet.trainTriplet(mnist, 1000, 128)  # 5000, 128 produces good results
    # siamese.saveModel()
    # siamese.loadModel()
    triplet.trainTripletForClassification(mnist, 1000, 128)

    # Test model
    embed = triplet.test_model(input1=mnist.test.images)
    embed = embed.reshape([-1, 2])
    visualize(embed, mnist_test_labels)
    triplet.computeAccuracy(mnist)


if __name__ == "__main__":
    sys.exit(int(main() or 0))

