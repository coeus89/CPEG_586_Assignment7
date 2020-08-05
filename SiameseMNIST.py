import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow.compat.v1 as tf
import matplotlib
import matplotlib.pyplot as plt
from Siamese import Siamese
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
    siamese = Siamese()
    siamese.trainSiamese(mnist, 1000, 128)  # 5000, 128 produces good results
    # siamese.saveModel()
    # siamese.loadModel()
    siamese.trainSiameseForClassification(mnist, 1000, 128)

    # Test model
    embed = siamese.test_model(input1=mnist.test.images)
    embed = embed.reshape([-1, 2])
    visualize(embed, mnist_test_labels)
    siamese.computeAccuracy(mnist)


# def main():
#    mnist = tf.keras.datasets.mnist
#    (x_train, y_train),(x_test,y_test) = mnist.load_data()
#    x_train, x_test = x_train / 255.0, x_test / 255.0
#
#    model = tf.keras.models.Sequential([
#        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dense(512,activation=tf.nn.relu),
#        tf.keras.layers.Dropout(0.2),
#        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#    ])
#
#    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#    model.fit(x_train, y_train,epochs=5)
#    model.evaluate(x_test,y_test)
#
if __name__ == "__main__":
    sys.exit(int(main() or 0))

