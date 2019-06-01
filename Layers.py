import tensorflow as tf
import numpy as np
def create_dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0, activation=None, batchnorm_enabled=False, dropout_keep_prob=-1, is_training=True):
    with tf.variable_scope(name) as scope:
        n_in = x.get_shape()[-1].value

        if w == None:
            kernel_shape = [n_in, output_dim]
            w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
            if (not tf.get_variable_scope().reuse):
                weight_decay = tf.multiply(tf.nn.l2_loss(w), l2_strength, name='w_loss')
                tf.add_to_collection(collection_name, weight_decay)


        bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))

        dense_o_b = tf.nn.bias_add(tf.matmul(x, w), bias)
        
       
        if not activation:
            dense_a = dense_o_b
        else:
            dense_a = activation(dense_o_b)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.nn.dropout(dense_a, dropout_keep_prob)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def create_conv(name, x, w = None, num_filters = 32, kernel_size = (8, 8), padding = 'VALID', stride = (4, 4),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation = tf.nn.relu, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):

    # initializer=orthogonal_initializer(np.sqrt(2))
    stride = [1, stride[0], stride[1], 1]
    kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

    with tf.variable_scope(name) as scope:
        
        with tf.name_scope('layer_weights'):
            w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
            if (not tf.get_variable_scope().reuse):
                weight_decay = tf.multiply(tf.nn.l2_loss(w), l2_strength, name='w_loss')
                tf.add_to_collection(collection_name, weight_decay)

        with tf.name_scope('layer_biases'):
            bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            conv_o_b = tf.nn.bias_add(conv, bias)
            conv_a = activation(conv_o_b)

            if dropout_keep_prob != -1:
                conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)
            else:
                conv_o_dr = conv_a

            conv_o = conv_o_dr
            if max_pool_enabled:
                conv_o = max_pool_2d(scope, conv_o_dr)
            
    return conv_o






def orthogonal_initializer(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # Orthogonal Initializer that uses SVD. The unused variables are just for passing in tensorflow
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init

def max_pool_2d(x, size=(2, 2)):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, size_x, size_y, 1], padding='VALID',
                          name='pooling')

def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o