#!-*-coding=utf-8-*-
import time
import tensorflow as tf
from my_code.qanet_layers import _linear
QUESTION_TYPE_MAP = {'ENTITY':[1, 0, 0], 'YES_NO':[0, 1, 0], 'DESCRIPTION':[0, 0, 1]}


def question_classification(q_embed_encoding, batch_size, max_q_len, hidden_size, filter_size=3, num_filters=96):
    """
    Joint learn reading comprehension and question type classification.
    :parameter q_embed_encoding: tensor of shape [batch_size * n_passage, q_len, hidden]
    :parameter batch_size: tensor of shape batch_size
    """
    with tf.variable_scope("Question_Type_classification") as scope:
        # rm dup question
        # q_embed_encoding shape [batch_size * n_passage, c_len, hidden]
        # no_dup_question_encodes shape [batch_size, c_len, hidden]
        no_dup_question_encodes = tf.reshape(q_embed_encoding, [batch_size, -1, max_q_len, hidden_size])[0:, 0, 0:, 0:]
        no_dup_question_encodes = tf.expand_dims(no_dup_question_encodes, -1)

        # Convolution Layer
        filter_shape = [filter_size, hidden_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            no_dup_question_encodes,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_q_len - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

        pooled = tf.reshape(pooled, [-1, num_filters])
        return _linear(pooled, 3, bias=True, scope=scope)


def process_feed_dict(feed_dict):
    feed_dict["question_type"] = []
    for data in feed_dict["raw_data"]:
        feed_dict["question_type"].append(QUESTION_TYPE_MAP.get(data["question_type"]))
    return feed_dict


if __name__ == "__main__":
    q = tf.zeros(shape=[32, 20, 96], dtype=tf.float32, name="test_input")
    result = question_classification(q, batch_size=16, max_q_len=20)
    print(result.shape)

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))