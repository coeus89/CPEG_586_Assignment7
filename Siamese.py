import tensorflow as tf
# import tensorflow.compat.v1 as tf
# from tensorflow import nn.conv2d
import os
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle

from tensorflow_core.contrib.learn.python.learn import trainable


# from tensorflow.examples.tutorials.mnist import input_data
# tf.compat.v1.disable_v2_behavior()

class Siamese(object):
    def __init__(self):
        # This is a TensorFlow v1 application so i have to include code to diable some v2 functionality.
        # tf.compat.v1.disable_v2_behavior()
        # tf.compat.v1.disable_eager_execution() # Only needed for v1 code with sessions ect...
        # ----set up place holders for inputs and labels for the siamese network---
        # two input placeholders for Siamese network

        self.tf_inputA = tf.placeholder(tf.float32, [None, 784], name='inputA')
        self.tf_inputB = tf.placeholder(tf.float32, [None, 784], name='inputB')

        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None, ], name='Y')
        self.tf_YOneHot = tf.placeholder(tf.float32, [None, 10], name='YoneHot')
        # outputs, loss function and training optimizer
        self.outputA, self.outputB = self.siameseNetwork()
        self.output = self.siameseNetworkWithCLassification()
        self.loss = self.contrastiveLoss()
        self.lossCrossEntropy = self.crossEntropyLoss()
        self.optimizer = self.optimizer_initializer()
        self.optimizerCrossEntropy = self.optimizer_initializer_crossEntropy()
        self.saver = tf.train.Saver()

        # Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def layer(self, tf_input, num_hidden_units, variable_name, trainable=True):
        # tf_input: batch_size x n_features
        # num_hidden_units: number of hidden units
        tf_weight_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        num_features = tf_input.get_shape()[1]  # tf_input.shape[0]????
        W = tf.get_variable(
            name=variable_name + 'W',
            dtype=tf.float32,
            shape=[num_features, num_hidden_units],
            initializer=tf_weight_initializer,
            trainable=trainable
        )
        b = tf.get_variable(
            name=variable_name + 'b',
            dtype=tf.float32,
            shape=[num_hidden_units],
            trainable=trainable
        )
        out = tf.add(tf.matmul(tf_input, W), b)
        return out

    def CNNLayer(self, tf_input, KernelSize, NumFeatureMaps, variable_name, trainable=True):
        tf_weight_initializer = tf.random_normal_initializer(mean=0, stdev=0.01)
        NumFeaturePrevLayer = tf_input.get_shape()[0]
        k = tf.get_variable(
            name=variable_name + "K",
            dtype=tf.float32,
            shape=[KernelSize, KernelSize, NumFeaturePrevLayer, NumFeatureMaps],
            initializer=tf_weight_initializer,
            trainable=trainable
        )
        b = tf.get_variable(
            name=variable_name + 'b',
            dtype=tf.float32,
            shape=[NumFeatureMaps],
            trainable=trainable
        )
        # do i need to do a for loop for the different kernels?

        out = tf.add(tf.nn.conv2d(tf_input, k, padding='VALID'), b)  # do i need to do strides in conv2d?

    def network(self, tf_input, trainable=True):
        # Setup FNN
        fc1 = self.layer(tf_input=tf_input, num_hidden_units=1024, trainable=trainable, variable_name='fc1')
        ac1 = tf.nn.relu(fc1)
        fc2 = self.layer(tf_input=ac1, num_hidden_units=1024, trainable=trainable, variable_name='fc2')
        ac2 = tf.nn.relu(fc2)
        fc3 = self.layer(tf_input=ac2, num_hidden_units=2, trainable=trainable, variable_name='fc3')
        return fc3

    def networkWithClassifcation(self, tf_input):
        # Setup FNN
        fc3 = self.network(tf_input, trainable=False)
        ac3 = tf.nn.relu(fc3)
        fc4 = self.layer(tf_input=ac3, num_hidden_units=80, trainable=True, variable_name='fc4')
        ac4 = tf.nn.relu(fc4)
        fc5 = self.layer(tf_input=ac4, num_hidden_units=10, trainable=True, variable_name='fc5')
        return fc5

    def siameseNetwork(self):
        with tf.variable_scope("siamese") as scope:
            outputA = self.network(self.tf_inputA)
            # share weights
            scope.reuse_variables()
            outputB = self.network(self.tf_inputB)
        return outputA, outputB

    def siameseNetworkWithCLassification(self):
        # Initialize Neural Network
        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE) as scope:
            output = self.networkWithClassifcation(self.tf_inputA)
        return output

    def contrastiveLoss(self, margin=5.0):
        with tf.variable_scope("siamese") as scope:
            labels = self.tf_Y
            # Euclidean Distance Squared
            dist = tf.pow(tf.subtract(self.outputA, self.outputB), 2, name='Dw')
            Dw = tf.reduce_sum(dist, 1)
            # add 1e-6 t increase the stability of calculating the gradients
            Dw2 = tf.sqrt(Dw + 1e-6, name='Dw2')
            # Loss Function
            lossSimilar = tf.multiply(labels, tf.pow(Dw2, 2), name='constrastiveLoss_1')
            lossDissimilar = tf.multiply(tf.subtract(1.0, labels), tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2),
                                         name='constrastiveLoss_2')
            loss = tf.reduce_mean(tf.add(lossSimilar, lossDissimilar), name='contrastiveLoss')
            return loss

    def crossEntropyLoss(self):
        labels = self.tf_YOneHot
        lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=labels))
        return lossd

    def optimizer_initializer(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0  # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        # optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer

    def optimizer_initializer_crossEntropy(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0  # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.lossCrossEntropy)
        return optimizer

    def trainSiamese(self, mnist, numIterations, batchSize=100):
        # Train the network
        for i in range(numIterations):
            input1, y1 = mnist.train.next_batch(batchSize)
            input2, y2 = mnist.train.next_batch(batchSize)
            label = (y1 == y2).astype('float')
            _, trainingLoss = self.sess.run([self.optimizer, self.loss],
                                            feed_dict={self.tf_inputA: input1, self.tf_inputB: input2,
                                                       self.tf_Y: label})
            if i % 50 == 0:
                print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def trainSiameseForClassification(self, mnist, numIterations, batchSize=10):
        # Train the network for classification via softmax
        for i in range(numIterations):
            input1, y1 = mnist.train.next_batch(batchSize)
            y1c = to_categorical(y1)  # convert labels to one hot
            labels = np.zeros(batchSize)
            _, trainingLoss = self.sess.run([self.optimizerCrossEntropy, self.lossCrossEntropy],
                                            feed_dict={self.tf_inputA: input1, self.tf_inputB: input1,
                                                       self.tf_YOneHot: y1c, self.tf_Y: labels})
            if i % 10 == 0:
                print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def computeAccuracy(self, mnist):
        labels = np.zeros(100)
        yonehot = np.zeros((100, 10))
        aout = self.sess.run(self.output,
                             feed_dict={self.tf_inputA: mnist.test.images, self.tf_inputB: mnist.test.images,
                                        self.tf_YOneHot: yonehot, self.tf_Y: labels})
        accuracyCount = 0
        testY = to_categorical(mnist.test.labels)
        for i in range(testY.shape[0]):
            maxIndex = aout[i].argmax(axis=0)
            if (testY[i, maxIndex] == 1):
                accuracyCount += 1
        print("Accuracy count = " + str(accuracyCount / testY.shape[0] * 100) + '%')

        # num_batches = mnist[0].shape[0]
        # # x_train1,y_train1 = tf.data.Dataset.from_tensor_slices(mnist[0]),tf.data.Dataset.from_tensor_slices(mnist[1])
        # # x_train2,y_train2 = tf.data.Dataset.from_tensor_slices(mnist[0]),tf.data.Dataset.from_tensor_slices(mnist[1])
        # x_train1,y_train1 = mnist[0],mnist[1]
        # x_train2,y_train2 = mnist[0],mnist[1]
        # for i in range(numIterations):
        #     # if (i == 0):
        #     #     x_train1,y_train1 = shuffle(mnist[0],mnist[1])
        #     #     x_train2,y_train2 = shuffle(mnist[0],mnist[1])
        #     iter1 = (i % num_batches) * batchSize
        #     #input1, y1 = x_train1.batch(batchSize), y_train1.batch(batchSize) 
        #     input1, y1 = x_train1[iter1:iter1 + batchSize], y_train1[iter1:iter1 + batchSize]
        #     input2, y2 = x_train2[iter1:iter1 + batchSize], y_train2[iter1:iter1 + batchSize]
        #     #input1, y1 = mnist.train.next_batch(batchSize)
        #     #input2, y2 = mnist.train.next_batch(batchSize)
        #     label = (y1 == y2).astype('float')
        #     _, trainingLoss = self.sess.run([self.optimizer, self.loss],feed_dict = {self.tf_inputA: input1, self.tf_inputB: input2,self.tf_Y: label})
        #     if i % 50 == 0:
        #         print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def test_model(self, input1):
        # Test the trained model
        output = self.sess.run(self.outputA, feed_dict={self.tf_inputA: input1})
        return output

    def saveModel(self):
        modelDir = "d:\\deeplearning\\trainedmodels\\"
        modelName = "SiameseJK"
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        # Save the latest trained models
        self.saver.save(self.sess, modelDir + modelName)

    def loadModel(self):
        # restore the trained model
        modelDir = "d:\\deeplearning\\trainedmodels\\"
        modelName = "SiameseJK"
        # assert os.path.exists(modelDir + modelName)
        self.saver.restore(self.sess, modelDir + modelName)

    def saveWeights(self, w1, b1, w2, b2):
        weights = {'w1': w1,
                   'b1': b1,
                   'w2': w2,
                   'b2': b2}
        with open('savedWeights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Weights saved successfully.')

