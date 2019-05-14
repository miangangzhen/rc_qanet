#!-*-coding=utf-8-*-
import time
import tensorflow as tf
import numpy as np
from my_code.qanet_layers import _linear
QUESTION_TYPE_MAP = {'ENTITY':[1, 0, 0], 'YES_NO':[0, 1, 0], 'DESCRIPTION':[0, 0, 1]}
YES_NO_ANSWER_MAP = {
    "Empty": np.array([1, 0, 0, 0, 0]),
    'Depends': np.array([0, 1, 0, 0, 0]),
    'No': np.array([0, 0, 1, 0, 0]),
    'Yes': np.array([0, 0, 0, 1, 0]),
    'No_Opinion': np.array([0, 0, 0, 0, 1])}
YES_NO_ANSWER_LIST = ["Empty", 'Depends', 'No', 'Yes', 'No_Opinion']


def question_classification(q_embed_encoding, batch_size, max_q_len, hidden_size, filter_size=3, num_filters=96):
    """
    Joint learn reading comprehension and question type classification.
    :parameter q_embed_encoding: tensor of shape [batch_size * n_passage, q_len, hidden]
    :parameter batch_size: tensor of shape batch_size
    """
    with tf.variable_scope("Question_Type_classification") as scope:
        # rm dup question
        # q_embed_encoding shape [batch_size * n_passage, q_len, hidden]
        # no_dup_question_encodes shape [batch_size, q_len, hidden]
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


def yes_no_classification(p_embed_encoding, batch_size, max_p_len, hidden_size, filter_size=3, num_filters=96, num_passage=2):
    with tf.variable_scope("YES_No_Answer_classification") as scope:
        # rm dup question
        # p_embed_encoding shape [batch_size * n_passage, p_len, hidden]
        # no_dup_question_encodes shape [batch_size, p_len*n_passage, hidden]
        passage_encodes = tf.reshape(p_embed_encoding, [batch_size, -1, hidden_size])
        passage_encodes = tf.expand_dims(passage_encodes, -1)

        # Convolution Layer
        filter_shape = [filter_size, hidden_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            passage_encodes,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_p_len * num_passage  - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

        pooled = tf.reshape(pooled, [-1, num_filters])
        return _linear(pooled, 5, bias=True, scope=scope)


# 在feed_dict里增加multi_task_learning的target
def process_feed_dict(feed_dict):
    feed_dict["question_type"] = []
    feed_dict["yes_no_answer"] = []
    for data in feed_dict["raw_data"]:
        # question type classification
        feed_dict["question_type"].append(QUESTION_TYPE_MAP.get(data["question_type"]))

        # yes_no_answer classification
        yes_no_target = np.zeros(5, dtype=int)
        if "yesno_answers" in data.keys():
            for item in data["yesno_answers"]:
                yes_no_target += YES_NO_ANSWER_MAP.get(item)
        else:
            yes_no_target += YES_NO_ANSWER_MAP.get("Empty")
        feed_dict["yes_no_answer"].append(yes_no_target)

    return feed_dict


from my_code.rouge_l_tensorflow import tf_rouge_l

def get_fun(max_length):
    def fun(x):
        start = x[0]
        end = x[1]
        range = tf.range(start, limit=end+1, dtype=tf.int32)
        scatter = tf.scatter_nd(tf.reshape(range, [-1, 1]), tf.ones_like(range), tf.reshape(max_length, [1]))
        return scatter
    return fun

def get_passage_mask(passage, start, end):
    start = tf.reshape(start, [-1, 1])
    end = tf.reshape(end, [-1, 1])
    start_end = tf.concat([start, end], axis=-1)
    start_end, _ = tf.nn.top_k(start_end, k=2)
    start_end = tf.reverse(start_end, axis=[-1])
    func = get_fun(tf.shape(passage)[1])
    scatters = tf.map_fn(func, start_end)
    return scatters


def get_mv_non_zero_func(max_length):
    def _mv_non_zero_to_first_place(tensor):
        boolean_mask = tf.logical_not(tf.equal(tensor, tf.zeros_like(tensor)))
        non_zero_values = tf.gather_nd(tensor, tf.where(boolean_mask))
        range = tf.range(0, limit=tf.shape(non_zero_values)[0], dtype=tf.int32)
        scatter = tf.scatter_nd(tf.reshape(range, [-1, 1]), non_zero_values, tf.reshape(max_length, [1]))
        return scatter
    return _mv_non_zero_to_first_place


def mv_non_zero_to_first_place(batch_tensor):
    func = get_mv_non_zero_func(tf.shape(batch_tensor)[1])
    return tf.map_fn(func, batch_tensor)


def get_rouge_loss(p_, logit_start, logit_end, label_start, label_end):
    """
    compute rouge loss
    :param p_: passage ids [batch_size*max_passage_num, sequence_length]
    :param logit_start: [batch_size, max_passage_num*max_p_len]
    :param logit_end: [batch_size, max_passage_num*max_p_len]
    :param label_start: [batch_size, max_passage_num*max_p_len]
    :param label_end: [batch_size, max_passage_num*max_p_len]
    :return:
    """
    batch_size = tf.shape(label_start)[0]
    p_ = tf.reshape(p_,[batch_size, -1])
    argmax_start = tf.cast(tf.argmax(logit_start, axis=-1), dtype=tf.int32)
    argmax_end = tf.cast(tf.argmax(logit_end, axis=-1), dtype=tf.int32)
    passage_logits_mask =  get_passage_mask(p_, argmax_start, argmax_end) * p_
    passage_logits_non_zero = mv_non_zero_to_first_place(passage_logits_mask)

    passage_label_mask = get_passage_mask(p_, label_start, label_end) * p_
    passage_label_non_zero = mv_non_zero_to_first_place(passage_label_mask)

    return -tf.log(tf.reduce_mean(tf_rouge_l(passage_logits_non_zero, passage_label_non_zero, 0)) + 1e-8)


if __name__ == "__main__":
    q = tf.zeros(shape=[32, 20, 96], dtype=tf.float32, name="test_input")
    # result = question_classification(q, batch_size=16, max_q_len=20, hidden_size=96)
    # print(result.shape)
    #
    # total_parameters = 0
    # for variable in tf.trainable_variables():
    #     shape = variable.get_shape()
    #     variable_parametes = 1
    #     for dim in shape:
    #         variable_parametes *= dim.value
    #     total_parameters += variable_parametes
    # print("Total number of trainable parameters: {}".format(total_parameters))


    result = yes_no_classification(q, batch_size=16, max_p_len=20, hidden_size=96)
    print(result.shape)

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
