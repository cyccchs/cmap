from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from multi_agent_4_cifarshnsnlnsgr_1_model import RecurrentAttentionModel
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.app.flags.DEFINE_string("STATE",                     'test',         "What do you like to do, [TRAIN] OR [TEST]")

tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-5, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("CAPACITY", 256, "CAPACITY")
tf.app.flags.DEFINE_integer("INPUT_IMAGE_LEN", 128, "INPUT_IMAGE_LEN")
tf.app.flags.DEFINE_integer("num_steps", 6000000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 8, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Hidden size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 10, "Number of glimpses.")
tf.app.flags.DEFINE_float("std", 0.22, "Gaussian std for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

tf.app.flags.DEFINE_integer("rnn_batch_size", 11, "soft attention batch size")
tf.app.flags.DEFINE_integer("lstm_size", 256, "soft attention lstm_size")
tf.app.flags.DEFINE_integer("Wa_size", 256, "soft attention Wa size")
tf.app.flags.DEFINE_integer("Wh_size", 256, "soft attention Wh size")

FLAGS = tf.app.flags.FLAGS

training_steps_per_epoch = 50000 // FLAGS.BATCH_SIZE

#80%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\2class_test_similar'
#65%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\2class_test'
#75%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\2class_train_test'
#85%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\2class_train_similar_test'30000
#8889%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\2class_train_similar_test'60000
#83%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\2class_even_test'60000   max size 128 80% INPUT_IMAGE_LEN 128 85%
#83%'C:\\Users\\Wes Kao\\.spyder-py3\\test\\3class_train_test'INPUT_IMAGE_LEN 128  INPUT_IMAGE_LEN 64 80%
ram = RecurrentAttentionModel(img_size=32, # MNIST: 28 * 28
                              Wa_size=FLAGS.Wa_size,
                              Wh_size=FLAGS.Wh_size,
                              rnn_batch_size=FLAGS.rnn_batch_size,
                              lstm_size=FLAGS.lstm_size,
                              pth_size=FLAGS.patch_window_size,
                              g_size=FLAGS.g_size,
                              l_size=FLAGS.l_size,
                              glimpse_output_size=FLAGS.glimpse_output_size,
                              loc_dim=2,   # (x,y)
                              std=FLAGS.std,
                              hidden_size=FLAGS.hidden_size,
                              num_glimpses=FLAGS.num_glimpses,
                              num_classes=10,
                              learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              min_learning_rate=FLAGS.min_learning_rate,
                              training_steps_per_epoch=training_steps_per_epoch,
                              max_gradient_norm=FLAGS.max_gradient_norm,
                              is_training=True)
if FLAGS.STATE == 'train':
    ram.train(FLAGS.num_steps, FLAGS.M, FLAGS.BATCH_SIZE, FLAGS.CAPACITY, FLAGS.INPUT_IMAGE_LEN)
elif FLAGS.STATE == 'test':
    ram.test(FLAGS.M, FLAGS.BATCH_SIZE, FLAGS.CAPACITY, FLAGS.INPUT_IMAGE_LEN, FLAGS.patch_window_size)
elif FLAGS.STATE == 'test2':
    ram.test2(FLAGS.M, FLAGS.BATCH_SIZE, FLAGS.patch_window_size)