import os
import glob
import random
from datetime import datetime
from zipfile import ZipFile
from os.path import basename

import numpy as np
import tensorflow as tf


print("Numpy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

random.seed(712)
np.random.seed(712)


# *****************************************Data Preparation*****************************************
TRAIN_DATA_DIR = "./dataset/train"
INFERENCE_DATA_DIR = "./dataset/validation"
TEST_SIZE = 0.2
IMG_SIZE = 32
BATCH_SIZE = 256
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
LAMBDA_BALANCE = 1.0
LAMBDA_REG = 0.0
NUM_THREADS = 4


def gen_data():

    img_paths = [p for p in glob.glob(TRAIN_DATA_DIR + '/*.jpg')]
    n_train = int(len(img_paths) * (1 - TEST_SIZE))
    train_img_paths = img_paths[:n_train]
    test_img_paths = img_paths[n_train:]
    inference_img_paths = [p for p in glob.glob(INFERENCE_DATA_DIR + '/*.jpg')]

    mean, stddev = [], []
    with open(TRAIN_DATA_DIR + '/label_train.txt', 'r') as file:
        for row in file:
            mean.append(float(row.split('\t')[0]))
            stddev.append(float(row.split('\t')[1]))
    train_mean, train_stddev = mean[:n_train], stddev[:n_train]
    test_mean, test_stddev = mean[n_train:], stddev[n_train:]

    inference_mean, inference_stddev = np.zeros(len(inference_img_paths)), np.zeros(len(inference_img_paths))

    train_data = integrate(train_img_paths, train_mean, train_stddev)
    test_data = integrate(test_img_paths, test_mean, test_stddev)
    inference_data = integrate(inference_img_paths, inference_mean, inference_stddev)
    print('Training data size: {}'.format(len(train_data)))
    print('Test data size: {}'.format(len(test_data)))
    print('Inference data size: {}'.format(len(inference_data)))

    return train_data, test_data, inference_data


def integrate(img_paths, mean, stddev):

    data = []
    for idx in range(len(img_paths)):
        data.append([img_paths[idx], mean[idx], stddev[idx]])

    return np.asarray(data)


### IMAGE READING PARSING ###

def read_img(img_path, is_training=False):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img = tf.image.resize_images(img_decoded, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0

    if is_training:
        """Data augmentation comes here"""
        img = tf.image.random_flip_left_right(img)

    return img


def parse_function(img_path, mean, stddev, is_training=False):

    img = read_img(img_path, is_training)

    return img, [mean], [stddev]


def parse_function_train(img_path, mean, stddev):

    return parse_function(img_path, mean, stddev, is_training=True)


def parse_function_test(img_path, mean, stddev):

    return parse_function(img_path, mean, stddev, is_training=False)


### DATA SERVING ###

class DataGenerator(object):

    def __init__(self, batch_size=1, num_threads=1,
                 train_shuffle=False, buffer_size=10000):
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.buffer_size = buffer_size

        # data sampling and spliting
        self.train_data, self.test_data, self.inference_data = gen_data()

        # build iterator
        self.train_set = self._build_data_set(self.train_data,
                                              parse_function_train,
                                              shuffle=train_shuffle)
        self.iterator = tf.data.Iterator.from_structure(self.train_set.output_types,
                                                        self.train_set.output_shapes)
        # for training
        self.train_init_op = self.iterator.make_initializer(self.train_set)
        self.next = self.iterator.get_next()
        self.num_train_batches = int(np.ceil(len(self.train_data) / batch_size))
        # for testing
        self.test_set = self._build_data_set(self.test_data, parse_function_test)
        self.test_init_op = self.iterator.make_initializer(self.test_set)
        self.num_test_batches = int(np.ceil(len(self.test_data) / batch_size))
        # for inference
        self.inference_set = self._build_data_set(self.inference_data, parse_function_test)
        self.inference_init_op = self.iterator.make_initializer(self.inference_set)
        self.num_inference_batches = int(np.ceil(len(self.inference_data) / batch_size))

    def _build_data_set(self, data, map_fn, shuffle=False):
        """
        Images are loaded from disk and processed batch by batch. Since our dataset
        is not that big, it would be faster if we load all the images into RAM once
        and read from their. I leave it for you guys to explore :)
        """
        img_path = tf.convert_to_tensor(data[:, 0], dtype=tf.string)
        mean = tf.convert_to_tensor(data[:, 1], dtype=tf.float64)
        stddev = tf.convert_to_tensor(data[:, 2], dtype=tf.float64)
        data = tf.data.Dataset.from_tensor_slices((img_path, mean, stddev))
        if shuffle:
            data = data.shuffle(buffer_size=self.buffer_size)
        data = data.map(map_fn, num_parallel_calls=self.num_threads)
        data = data.batch(self.batch_size)
        data = data.prefetch(self.num_threads)
        return data


# *****************************************Model Architecture*****************************************
class MLP(object):

    def __init__(self, training=False):

        self.x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
        self.y1 = tf.placeholder(tf.float32, [None, 1])
        self.y2 = tf.placeholder(tf.float32, [None, 1])

        net = self._encoder(self.x)

        with tf.variable_scope('regression'):
            self.mean = tf.layers.dense(net, 1, name='mean')
            self.stddev = tf.layers.dense(net, 1, name='stddev')

        if training:
            self.loss, self.train_op = self._loss_fn()

    def _encoder(self, input, name='encoder'):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            net = tf.layers.flatten(input)
            net = tf.layers.dense(net, units=300, activation=tf.nn.relu)
            net = tf.layers.dense(net, units=300, activation=tf.nn.relu)

            return net

    def _loss_fn(self):

        trained_vars = tf.trainable_variables()

        error_mean = tf.sqrt(tf.reduce_mean((self.y1 - self.mean) ** 2))
        error_stddev = tf.sqrt(tf.reduce_mean((self.y2 - self.stddev) ** 2))

        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in trained_vars
                           if 'bias' not in v.name])
        loss = error_mean + LAMBDA_BALANCE * error_stddev + LAMBDA_REG * l2_reg

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE,
                                           beta1=0.9,
                                           beta2=0.99,
                                           epsilon=1e-8)
        train_op = optimizer.minimize(loss)#, global_step, var_list=trained_vars)

        return loss, train_op


# *****************************************Training, Test, and Inference*****************************************
generator = DataGenerator(batch_size=BATCH_SIZE, num_threads=NUM_THREADS, train_shuffle=True, buffer_size=10000)
model = MLP(training=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\n{} Epoch: {}/{}".format(datetime.now(), epoch, NUM_EPOCHS))

        # **********************Training**********************
        sum_loss = 0.
        sess.run(generator.train_init_op)
        for step in range(generator.num_train_batches):
            batch_img, batch_mean, batch_stddev = sess.run(generator.next)
            _, loss = sess.run([model.train_op, model.loss], feed_dict={model.x: batch_img,
                                                                        model.y1: batch_mean,
                                                                        model.y2: batch_stddev})
            sum_loss += loss
        print('Training loss: {:.6f}'.format(sum_loss))

        # **********************Test**********************
        pred_means = []
        pred_stddevs = []
        true_means = []
        true_stddevs = []
        sum_loss = 0.
        sess.run(generator.test_init_op)
        for step in range(generator.num_test_batches):
            batch_img, batch_mean, batch_stddev = sess.run(generator.next)
            pred_mean, pred_stddev, loss = sess.run([model.mean, model.stddev, model.loss], feed_dict={model.x: batch_img,
                                                                                                       model.y1: batch_mean,
                                                                                                       model.y2: batch_stddev})
            sum_loss += loss
            pred_means.extend(pred_mean.ravel().tolist())
            pred_stddevs.extend(pred_stddev.ravel().tolist())
            true_means.extend(batch_mean.ravel().tolist())
            true_stddevs.extend(batch_stddev.ravel().tolist())
        pred_means = np.asarray(pred_means)
        pred_stddevs = np.asarray(pred_stddevs)
        true_means = np.asarray(true_means)
        true_stddevs = np.asarray(true_stddevs)
        print('Test loss: {:.6f}'.format(sum_loss))
        print('Test mean RMSE: {:.6f}'.format(np.sqrt(np.mean((pred_means - true_means) ** 2))))
        print('Test stddev RMSE: {:.6f}'.format(np.sqrt(np.mean((pred_stddevs - true_stddevs) ** 2))))

        # **********************Inference**********************
        if not os.path.exists('./submission'):
            os.makedirs('./submission')
        inference_means = []
        inference_stddevs = []
        sum_loss = 0.
        sess.run(generator.inference_init_op)
        for step in range(generator.num_inference_batches):
            batch_img, batch_mean, batch_stddev = sess.run(generator.next)
            pred_mean, pred_stddev, _ = sess.run([model.mean, model.stddev, model.loss], feed_dict={model.x: batch_img,
                                                                                                    model.y1: batch_mean,
                                                                                                    model.y2: batch_stddev})
            inference_means.extend(pred_mean.ravel().tolist())
            inference_stddevs.extend(pred_stddev.ravel().tolist())
        inference_means = np.asarray(inference_means)
        inference_stddevs = np.asarray(inference_stddevs)

        # Output
        with open('./submission/prediction.txt', 'w') as file:
            for idx in range(len(inference_means)):
                file.write(str(inference_means[idx]))
                file.write('\t')
                file.write(str(inference_stddevs[idx]))
                file.write('\n')
        ZipFile('./submission/prediction.zip', 'w').write('./submission/prediction.txt', basename('prediction.txt'))
        print('Inference results saved!')