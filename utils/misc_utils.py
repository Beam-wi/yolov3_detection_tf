# coding: utf-8

import numpy as np
import tensorflow as tf
import random
import os

from tensorflow.core.framework import summary_pb2


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def update_dict(ori_dict, new_dict):
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict


def list_add(ori_list, new_list):
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with open(weights_file, "rb") as fp:
        a = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def config_learning_rate(config, global_step, lr_decay_freq):
    if config.model_set["lr_type"] == 'exponential':
        lr_tmp = tf.train.exponential_decay(config.model_set["learning_rate_init"], global_step, lr_decay_freq,
                                            config.model_set["lr_decay_factor"], staircase=True, name='exponential_learning_rate')

        return tf.maximum(lr_tmp, config.model_set["lr_lower_bound"])
    elif config.model_set["lr_type"] == 'fixed':
        return tf.convert_to_tensor(config.model_set["learning_rate_init"], name='fixed_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')


def average_gradients(tower_grads):
    '''
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    :param tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    :return: List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    '''

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_background_lines(background_dir):
    def get_chile_path(father_path):
        child_paths = []
        for name in os.listdir(father_path):
            child_paths.append(os.path.join(father_path, name))
        return child_paths

    child_paths = get_chile_path(background_dir)
    background_lines = []
    for child_path in child_paths:
        for name in os.listdir(child_path):
            if name.split(".")[-1] == "jpg":
                background_lines.append(os.path.join(child_path, name))
            else:
                pass
    return background_lines
