# -*- coding: utf-8 -*-
"""Created on Wed Apr 18 23:47:52 201i8

@author: sony
"""
# import necessary libraries
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
 Load Input Image and segmented image.
 Convert it to 512*512*512*1 size
'''

# Load original MRI CT Scan File
test_raw = loadmat('MATlab/CT-03.mat')['data']
# Load manually done image annotations
test_mask_raw = loadmat('MATlab/seg-03.mat')['data']
# input shape: (batch, depth, height, width, channels)
# resizing both original input and segmented image to 512*512*512*1
test = np.asarray(test_raw).reshape((1, 512, 512, 492, 1))
test_mask = np.asarray(test_mask_raw).reshape((1, 512, 512, 492, 1))
train1 = test.copy()
train1.resize(1, 512, 512, 512, 1)
trainlab1 = test_mask.copy()
trainlab1.resize(1, 512, 512, 512, 1)

test_raw1 = loadmat('MATlab/CT-04.mat')['data']
test_mask_raw1 = loadmat('MATlab/seg-04.mat')['data']
print("Loaded test file shape: ", test_raw1.shape)
print("Loaded test ground truth shape: ", test_mask_raw1.shape)
# input shape: (batch, depth, height, width, channels)
test1 = np.asarray(test_raw1).reshape((1, 512, 512, 531, 1))
test_mask1 = np.asarray(test_mask_raw1).reshape((1, 512, 512, 531, 1))
train2 = test1.copy()
train2.resize(1, 512, 512, 512, 1)
trainlab2 = test_mask1.copy()
trainlab2.resize(1, 512, 512, 512, 1)

test_raw2 = loadmat('MATlab/CT-05.mat')['data']
test_mask_raw2 = loadmat('MATlab/seg-05.mat')['data']
# input shape: (batch, depth, height, width, channels)
print("Loaded test file shape: ", test_raw2.shape)
print("Loaded test ground truth shape: ", test_mask_raw2.shape)
test2 = np.asarray(test_raw2).reshape((1, 512, 512, 501, 1))
test_mask2 = np.asarray(test_mask_raw2).reshape((1, 512, 512, 501, 1))
train3 = test2.copy()
train3.resize(1, 512, 512, 512, 1)
trainlab3 = test_mask2.copy()
trainlab3.resize(1, 512, 512, 512, 1)

test_raw3 = loadmat('MATlab/CT-06.mat')['data']
test_mask_raw3 = loadmat('MATlab/seg-06.mat')['data']
# input shape: (batch, depth, height, width, channels)
print("Loaded test file shape: ", test_raw3.shape)
print("Loaded test ground truth shape: ", test_mask_raw3.shape)
test3 = np.asarray(test_raw3).reshape((1, 512, 512, 419, 1))
test_mask3 = np.asarray(test_mask_raw3).reshape((1, 512, 512, 419, 1))
train4 = test3.copy()
train4.resize(1, 512, 512, 512, 1)
trainlab4 = test_mask3.copy()
trainlab4.resize(1, 512, 512, 512, 1)

test_raw4 = loadmat('MATlab/CT-07.mat')['data']
test_mask_raw4 = loadmat('MATlab/seg-07.mat')['data']
# input shape: (batch, depth, height, width, channels)
print("Loaded test file shape: ", test_raw4.shape)
print("Loaded test ground truth shape: ", test_mask_raw4.shape)
test4 = np.asarray(test_raw4).reshape((1, 512, 512, 501, 1))
test_mask4 = np.asarray(test_mask_raw4).reshape((1, 512, 512, 501, 1))
train5 = test4.copy()
train5.resize(1, 512, 512, 512, 1)
trainlab5 = test_mask4.copy()
trainlab5.resize(1, 512, 512, 512, 1)

test_raw5 = loadmat('MATlab/CT-08.mat')['data']
test_mask_raw5 = loadmat('MATlab/seg-08.mat')['data']
# input shape: (batch, depth, height, width, channels)
print("Loaded test file shape: ", test_raw5.shape)
print("Loaded test ground truth shape: ", test_mask_raw5.shape)
test5 = np.asarray(test_raw5).reshape((1, 512, 512, 467, 1))
test_mask5 = np.asarray(test_mask_raw5).reshape((1, 512, 512, 467, 1))
train6 = test5.copy()
train6.resize(1, 512, 512, 512, 1)
trainlab6 = test_mask5.copy()
trainlab6.resize(1, 512, 512, 512, 1)

test_raw6 = loadmat('MATlab/CT-09.mat')['data']
test_mask_raw6 = loadmat('MATlab/seg-09.mat')['data']
# input shape: (batch, depth, height, width, channels)
print("Loaded test file shape: ", test_raw6.shape)
print("Loaded test ground truth shape: ", test_mask_raw6.shape)
test6 = np.asarray(test_raw6).reshape((1, 512, 512, 544, 1))
test_mask6 = np.asarray(test_mask_raw6).reshape((1, 512, 512, 544, 1))
train7 = test6.copy()
train7.resize(1, 512, 512, 512, 1)
trainlab7 = test_mask6.copy()
trainlab7.resize(1, 512, 512, 512, 1)

'''
Stack Inputs for training
'''
# Stack Original Inputs
train_test = tf.concat([train1, train2, train3, train4, train5], 0)
# Stack Segmented Inputs
train_label = tf.concat([trainlab1, trainlab2, trainlab3, trainlab4, trainlab5], 0)

# convert tensor to numpy array
train_test = tf.Session().run(train_test)
train_label = tf.Session().run(train_label)

float_X_training = train_test.astype(np.float32)
float_Y_training = train_label.astype(np.int32)

print("Training size : ", float_X_training.shape)
print("Label size : ", float_Y_training.shape)

'''
Make test Dataset
Do same processing as done for training

I have just tested for 2 images
'''

test_raw7 = loadmat('MATlab/CT-10.mat')['data']
test_mask_raw7 = loadmat('MATlab/seg-10.mat')['data']
# input shape: (batch, depth, height, width, channels)
print("Loaded test file shape: ", test_raw7.shape)
test7 = np.asarray(test_raw7).reshape((1, 512, 512, 518, 1))
test_mask7 = np.asarray(test_mask_raw7).reshape((1, 512, 512, 518, 1))
train8 = test7.copy()
train8.resize(1, 512, 512, 512, 1)
trainlab8 = test_mask7.copy()
trainlab8.resize(1, 512, 512, 512, 1)

testing_raw = loadmat('MATlab/CT-11.mat')['data']
testing_mask_raw = loadmat('MATlab/seg-11.mat')['data']
print("Loaded test file shape: ", testing_raw.shape)
testing = np.asarray(testing_raw).reshape((1, 512, 512, 539, 1))
testing_mask = np.asarray(testing_mask_raw).reshape((1, 512, 512, 539, 1))
X_test1 = testing.copy()
X_test1.resize(1, 512, 512, 512, 1)
Y_test1 = testing_mask.copy()
Y_test1.resize(1, 512, 512, 512, 1)

'''
Stack inputs for testing
'''
X_bundle_test = tf.concat([train8, X_test1], 0)
Y_bundle_test = tf.concat([trainlab8, Y_test1], 0)

new_graph = tf.Graph()

X_bundle_test = tf.Session().run(X_bundle_test)
Y_bundle_test = tf.Session().run(Y_bundle_test)

float_X_testing = X_bundle_test.astype(np.float32)
float_Y_testing = Y_bundle_test.astype(np.int32)

print("Testing size : ", float_X_testing.shape)
print("Testing Label size : ", float_Y_testing.shape)


def CNN_UNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    original_shape = x.get_shape()

    # TODO: Layer 1 sublayer 1
    W1_1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 1, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W1_1, strides=[1, 1, 1, 1, 1], padding='SAME')
    b1_1 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b1_1)
    print("layer 1_1 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 1 sublayer 2
    W1_2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 3, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W1_2, strides=[1, 1, 1, 1, 1], padding='SAME')
    b1_2 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b1_2)
    print("layer 1_2 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 1x530x512x512x8. Output = 1x265x256x256x8.
    x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    layer1 = x
    print("layer 1 shape:", x.get_shape())

    # TODO: Layer 2 sublayer 1
    W2_1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 3, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W2_1, strides=[1, 1, 1, 1, 1], padding='SAME')
    b2_1 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b2_1)
    print("layer 2_1 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    W2_2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 3, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W2_2, strides=[1, 1, 1, 1, 1], padding='SAME')
    b2_2 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b2_2)
    print("layer 2_2 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 1x265x256x256x16. Output = 1x133x128x128x16
    x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    layer2 = x

    print("layer 2 shape:", x.get_shape())

    # TODO: Layer 3 sublayer 1
    W3_1 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 3, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W3_1, strides=[1, 1, 1, 1, 1], padding='SAME')
    b3_1 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b3_1)
    print("layer 3_1 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Layer 3 sublayer 2
    W3_2 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 3, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W3_2, strides=[1, 1, 1, 1, 1], padding='SAME')
    b3_2 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b3_2)
    print("layer 3_2 shape:", x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    # TODO: Layer 3 sublayer 3
    W3_3 = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 3, 3), mean=mu, stddev=sigma))
    x = tf.nn.conv3d(x, W3_3, strides=[1, 1, 1, 1, 1], padding='SAME')
    b3_3 = tf.Variable(tf.zeros(3))
    x = tf.nn.bias_add(x, b3_3)
    print("layer 3_3 shape:", x.get_shape())
    layer3 = x

    # TODO: Activation.
    x = tf.nn.relu(x)

    # FINAL dOWNSAMPLING LAYER (2 classes)
    final_down_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 1, 3, 2), mean=mu, stddev=sigma))
    final_down_b = tf.Variable(tf.zeros(2))
    x = tf.nn.conv3d(x, final_down_W, strides=[1, 1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, final_down_b)
    final_down_layer = x
    print("final layer shape:", final_down_layer.get_shape())

    '''
        Upsampling : Denconvolution
    '''
    # now to upscale to actual image size
    # TODO:1st Upsample Layer output size of DownSampling layer3
    deconv_shape1 = layer3.get_shape()
    x = tf.contrib.layers.conv3d_transpose(x, deconv_shape1[4], kernel_size=[4, 4, 4], stride=[2, 2, 2])
    print("1st Up sample layer:", x.get_shape())

    # now to upscale to actual image size
    # TODO:2nd Upsample Layer output size of DownSampling layer2
    deconv_shape2 = layer2.get_shape()
    x = tf.contrib.layers.conv3d_transpose(x, deconv_shape2[4], kernel_size=[4, 4, 4], stride=[2, 2, 2])
    print("2nd down sample layer:", x.get_shape())

    # TODO:3rd Upsample Layer output size actual image size
    x = tf.contrib.layers.conv3d_transpose(x, 2, kernel_size=[4, 4, 4], stride=[1, 1, 1])
    logits = x
    print("Final up sample layer:", x.get_shape())

    # Returns the index with the largest value across axis of a tensor. will output either 0 or 1 index. Hence binary mask
    annotation_pred = tf.argmax(logits, axis=4)
    print("Output shape : ", annotation_pred.get_shape())
    # argmax will output array which will be of shape 1x530x512x512
    # So use expand_dim to make it  1x530x512x512x1

    return tf.expand_dims(annotation_pred, axis=4), logits

# make a placeholder in X and Y
x = tf.placeholder(tf.float32, (1, 512, 512, 512, 1))
y = tf.placeholder(tf.int64, (1, 512, 512, 512, 1))
pred_annotations, logits = CNN_UNet(x)
# loss function crossentropy loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(y, squeeze_dims=[4]))
loss_operation = tf.reduce_mean(cross_entropy)
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.009)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(pred_annotations, y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
prediction = pred_annotations

EPOCHS = 20

'''
for testing 
'''


def evaluate(X_data, y_data):
    sess = tf.get_default_session()
    BATCH_SIZE = 1
    num_test = len(X_data)
    total_accuracy = 0
    for offset_test in range(0, num_test, BATCH_SIZE):
        end_test = offset_test + BATCH_SIZE
        X_test_batch, Y_test_batch = X_data[offset_test:end_test], y_data[offset_test:end_test]
        pred, accuracy = sess.run([prediction, accuracy_operation], feed_dict={x: X_test_batch, y: Y_test_batch})
        total_accuracy = total_accuracy + accuracy
    return pred, (total_accuracy) / num_test


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(float_X_training)
    BATCH_SIZE = 2

    print("Training...")
    for i in range(EPOCHS):
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            X_batch, Y_batch = float_X_training[offset:end], float_Y_training[offset:end]
            sess.run(training_operation, feed_dict={x: X_batch, y: Y_batch})
        pred_image, validation_accuracy = evaluate(float_X_testing, float_Y_testing)
        print("EPOCH : ", i)
        print("Validation Accuracy = ", validation_accuracy)
    m = dict()
    m['data'] = pred_image
    savemat('temp', m, do_compression=True)
