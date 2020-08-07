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
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    mnist2 = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist2.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # Get a numpy list of each class from training images
    num_classes = 10
    # class_Images = np.zeros((num_classes, int(train_images.shape[0] / num_classes), train_images.shape[1], train_images.shape[2]))
    class_Images = np.empty((num_classes), dtype=object)
    for j in range(num_classes):
        temp = []
        for i in range(len(train_labels)):
            if train_labels[i] == j:
                temp.append(train_images[i])
        class_Images[j] = np.array(temp)
    # class_Images = class_Images.astype(np.float32, casting='unsafe')
    triplet = Triplet()
    triplet.trainTriplet(train_images, train_labels, class_Images, 1000, 128)
    # siamese.saveModel()
    # siamese.loadModel()
    triplet.trainTripletForClassification(train_images, train_labels, 1000, 128)

    # Test model
    embed = triplet.test_model(input1=test_images)
    embed = embed.reshape([-1, 2])
    visualize(embed, test_labels)
    triplet.computeAccuracy(test_images, test_labels)


if __name__ == "__main__":
    sys.exit(int(main() or 0))

