import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

class func():
    def __init__(self, tensors, name, shape):
        if tensors is not None:
            self.keys, self.values = tensors
            self.n = self.keys.get_shape()[1].value/2
        else:
            tmp_k = tf.Variable( tf.random_normal(shape, stddev=0.3), name=name+'_keys', dtype=tf.float32 ) 
            self.n = shape[1]
            self.keys = tf.concat([tf.cos(tmp_k), tf.sin(tmp_k)], axis=1)
            self.values = tf.Variable( tf.random_normal( [shape[0], 1], stddev=0.3), name=name+'_values', dtype=tf.float32 )

    def dot(self, f2):
        keys_1 = self.keys
        values_1 = self.values 
        keys_2 = f2.keys
        values_2 = f2.values

        all_keys = tf.concat([ keys_1, keys_2 ], axis=0)
        query_r_1 = tf.matmul(tf.nn.softmax(tf.matmul(all_keys, keys_1, transpose_b=True)/self.n), values_1)
        query_r_2 = tf.matmul(tf.nn.softmax(tf.matmul(all_keys, keys_2, transpose_b=True)/self.n), values_2)

        query_r = query_r_1*query_r_2
        return tf.reduce_sum(query_r)


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.reshape(point_cloud, [-1, 3])

    grid_num = 96
    super_vec_num = 256

    w_encode = tf.Variable( tf.random_normal([3, grid_num], stddev=0.2), name='w_encode', dtype=tf.float32 )
    
    super_vecs = []
    for i in range(super_vec_num):
        super_vecs.append( func(None, 'super_vec_' + str(i), [1024, grid_num]) )

    input_image_split = tf.split(input_image, axis=0)
    items = [] 
    for img in input_image_split:
        keys = tf.matmul( tf.squeeze(img), w_encode )
        keys = tf.concat([tf.cos(keys), tf.sin(keys)], axis=1)
        values = tf.ones([num_point, 1], dtype=tf.float32)
        super_vec_in = func([keys, values], None, None)
        tmps = [] 
        for sv in super_vecs:
            tmps.append(super_vec_in.dot(sv))
        tmps = tf.concat(tmps, axis=0)
        items.append(tmps)
    net = tf.stack(items, axis=0)

    print(net.get_shape())

    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
