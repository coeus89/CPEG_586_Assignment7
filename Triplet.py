import tensorflow as tf
#import tensorflow.compat.v1 as tf
#from tensorflow import nn.conv2d
import os
from sklearn.utils import shuffle
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
import random

from tensorflow_core.contrib.learn.python.learn import trainable
#from tensorflow.examples.tutorials.mnist import input_data
#tf.compat.v1.disable_v2_behavior()

class Triplet(object):
    def __init__(self):
        # This is a TensorFlow v1 application so i have to include code to diable some v2 functionality.
        # tf.compat.v1.disable_v2_behavior()
        # tf.compat.v1.disable_eager_execution() # Only needed for v1 code with sessions ect...
        #----set up place holders for inputs and labels for the siamese network---
        # two input placeholders for Siamese network

        self.tf_inputA = tf.placeholder(tf.float32, [None, 784], name='inputA')
        self.tf_inputB = tf.placeholder(tf.float32, [None, 784], name='inputB')
        self.tf_inputC = tf.placeholder(tf.float32, [None, 784], name='inputC')

        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None,], name='Y')
        self.tf_YOneHot = tf.placeholder(tf.float32, [None, 10], name='YoneHot')
        # outputs, loss function and training optimizer
        self.outputA1, self.outputB1, self.outputC1 = self.tripletNetwork()
        self.outputT = self.tripletNetworkWithCLassification()
        self.loss = self.tripletContrastiveLoss()
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
        tf_weight_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01)
        num_features = tf_input.get_shape()[1]  # tf_input.shape[0]????
        W = tf.get_variable(
            name = variable_name + 'W',
            dtype = tf.float32,
            shape=[num_features, num_hidden_units],
            initializer = tf_weight_initializer,
            trainable = trainable
        )
        b = tf.get_variable(
            name = variable_name + 'b',
            dtype = tf.float32,
            shape=[num_hidden_units],
            trainable = trainable
        )
        out = tf.add(tf.matmul(tf_input,W), b)
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

        out = tf.add(tf.nn.conv2d(tf_input, k, padding='VALID'), b)  # do i need to do strides in conv2d? - NOT HERE

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

    def tripletNetwork(self):
        with tf.variable_scope("triplet") as scope:
            outputA = self.network(self.tf_inputA)
            # share weights
            scope.reuse_variables()
            outputB = self.network(self.tf_inputB)
            outputC = self.network(self.tf_inputC)
        return outputA, outputB, outputC

    def tripletContrastiveLoss(self, margin=0.2):
        with tf.variable_scope("triplet") as scope:
            labels = self.tf_Y
            # distPlus = tf.pow(tf.subtract(self.outputA1, self.outputB1), 2, name='distancePlus')  # Anchor - Positive
            # distMinus = tf.pow(tf.subtract(self.outputA1, self.outputC1), 2, name='distanceMinus')  # Anchor - Negative
            # lossFN = tf.add(tf.subtract(distPlus, distMinus), margin, name='totalDistance')  # Margin prevents trivial outputs
            # loss = tf.maximum(tf.reduce_sum(tf.multiply(lossFN, -1)), 0, name='tripletContLoss')

            anchor = self.outputA1
            pos = self.outputB1
            neg = self.outputC1
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1, name='distancePlus')
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1, name='distanceMinus')
            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin, name='totalDistance')
            loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
            # distPlusNP = distPlus.eval(sess=self.sess)
            # distMinusNP = distMinus.eval(sess=self.sess)
            # lossFNNP = lossFN.eval(sess=self.sess)
            # lossNP = loss.eval(sess=self.sess)
            # temp = ''
            return loss

    def tripletNetworkWithCLassification(self):
        # Initialize Neural Network
        with tf.variable_scope("triplet", reuse=tf.AUTO_REUSE) as scope:
            output = self.networkWithClassifcation(self.tf_inputA)
        return output

    def crossEntropyLoss(self):
        labels = self.tf_YOneHot
        lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputT, labels=labels))
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

    def trainTriplet(self, trainImages, trainLabels, class_Images, iterations, batchSize=100):
        # Train the network
        timages = trainImages.reshape((trainImages.shape[0], trainImages.shape[1] * trainImages.shape[2]))
        tlabels = trainLabels
        for i in range(iterations):
            timages, tlabels = shuffle(timages, tlabels)
            input1, y1 = timages[0:batchSize], tlabels[0:batchSize]
            input3, y3 = timages[batchSize:batchSize * 2], tlabels[batchSize:batchSize * 2]
            temp = []
            for j in range(batchSize):
                y_val = y1[j]
                rng = random.randint(0, len(class_Images[y_val]) - 1)
                imageList = class_Images[y_val]
                temp.append(imageList[rng])
            input2 = np.array(temp).reshape((batchSize, len(temp[1]) * len(temp[2])))
            label = (y1 == y3).astype('float')
            _, trainingLoss = self.sess.run([self.optimizer, self.loss], feed_dict={self.tf_inputA: input1, self.tf_inputB: input2, self.tf_inputC: input3, self.tf_Y: label})
            if i % 50 == 0:
                print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def trainTripletForClassification(self, test_images, test_labels, numIterations, batchSize=10):
        # Train the network for classification via softmax
        timages = test_images
        tlabels = test_labels
        for i in range(numIterations):
            timages, tlabels = shuffle(timages, tlabels)
            input1, y1 = timages[0:batchSize].reshape((batchSize, timages.shape[1] * timages.shape[2])), tlabels[
                                                                                                         0:batchSize]
            y1c = to_categorical(y1)  # convert labels to one hot
            labels = np.zeros(batchSize)
            _, trainingLoss = self.sess.run([self.optimizerCrossEntropy, self.lossCrossEntropy], feed_dict={self.tf_inputA: input1, self.tf_YOneHot: y1c, self.tf_Y: labels})
            if i % 10 == 0:
                print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def computeAccuracy(self, test_images, test_labels):
        labels = np.zeros(test_images.shape[0])
        yonehot = np.zeros((test_images.shape[0], 10))
        test_images1 = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
        aout = self.sess.run(self.outputT, feed_dict={self.tf_inputA: test_images1, self.tf_YOneHot: yonehot, self.tf_Y: labels})
        # aout = tf.nn.softmax(aout)
        accuracyCount = 0
        testY = to_categorical(test_labels)
        for i in range(testY.shape[0]):
            maxIndex = aout[i].argmax(axis=0)
            if testY[i, maxIndex] == 1:
                accuracyCount += 1
        print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%')

    def test_model(self, input1):
        # Test the trained model
        input1 = input1.reshape((input1.shape[0], input1.shape[1] * input1.shape[2]))
        output = self.sess.run(self.outputA1, feed_dict = {self.tf_inputA: input1})
        return output

    def saveModel(self):
        modelDir = "d:\\deeplearning\\trainedmodels\\"
        modelName = "TripletJK"
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
        # Save the latest trained models
        self.saver.save(self.sess, modelDir + modelName)

    def loadModel(self):
        # restore the trained model
        modelDir = "d:\\deeplearning\\trainedmodels\\"
        modelName = "TripletJK"
        #assert os.path.exists(modelDir + modelName)
        self.saver.restore(self.sess, modelDir + modelName)

    def saveWeights(self, w1, b1, w2, b2):
        weights = { 'w1': w1,
                    'b1': b1,
                    'w2': w2,
                    'b2': b2}
        with open('savedWeights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Weights saved successfully.')

