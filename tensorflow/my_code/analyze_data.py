#-*-coding=utf-8-*-
import os
import time

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from tqdm import tqdm

from my_code.my_prepare import load_dataset, Param

"""
官方给出的passage_max_len=500，太长了，非常容易爆显存
分析一下数据集，看看有没有办法处理
"""
ROOT_PATH = "../data/"

TRAIN_SEARCH_PATH = "train_preprocessed/trainset/search.train.json"
TRAIN_ZD_PATH = "train_preprocessed/trainset/zhidao.train.json"

DEV_SEARCH_PATH = "dev_preprocessed/devset/search.dev.json"
DEV_ZD_PATH = "dev_preprocessed/devset/zhidao.dev.json"

TEST_SEARCH_PATH = "test1_preprocessed/test1set/search.test1.json"
TEST_ZD_PATH = "test1_preprocessed/test1set/zhidao.test1.json"


def get_default_param():
    return Param(["no use"],
                  dev_files=[],
                  test_files=[],
                  vocab_dir="no use",
                  model_dir="no use",
                  result_dir="no use",
                  summary_dir="no use",
                  max_p_num=5,
                  max_p_len=500,
                  max_q_len=60,
                  embed_size=100,
                  use_embd=True)


def statistic_plot(path):
    tf.logging.warn(path)

    param = get_default_param()

    # 统计question type分布
    question_type_dict = defaultdict(int)
    # 统计question len分布
    question_len_list = []
    # 统计passage len分布
    passage_len_list = []
    # 统计passage num分布
    passage_num_list = []
    # 统计answer len分布
    answer_len_list = []
    # 统计answer passage分布
    answer_passage_dict = defaultdict(int)

    i = 0
    for sample in tqdm(load_dataset(os.path.join(ROOT_PATH, path), param)):
        if i == 0:
            tf.logging.info(sample.keys())

        tf.logging.info("question type: ")
        tf.logging.info(sample["question_type"])
        question_type_dict[sample["question_type"]] += 1
        tf.logging.info("question tokens: ")
        tf.logging.info(sample["question_tokens"])
        question_len_list.append(len(sample["question_tokens"]))

        tf.logging.info("passage: ")
        tf.logging.info(sample["passages"])
        passage_len_list.extend([len(x["passage_tokens"]) for x in sample["passages"]])
        passage_num_list.append(len(sample["passages"]))

        if "answers" in sample.keys():
            tf.logging.info("answer: ")
            tf.logging.info(sample["answers"])
            pass
        if "answer_spans" in sample.keys():
            tf.logging.info("answer span: ")
            tf.logging.info(sample["answer_spans"])
            if len(sample['answer_spans']) != 0:
                answer_len_list.append(sample["answer_spans"][0][1] - sample["answer_spans"][0][0])
        if 'answer_passages' in sample.keys():
            tf.logging.info("answer passages: ")
            tf.logging.info(sample["answer_passages"])
            for x in sample["answer_passages"]:
                answer_passage_dict[x] += 1

        i += 1
        # if i == 100:
        #     break

    tf.logging.warn("question type分布")
    tf.logging.warn(question_type_dict)
    tf.logging.warn("answer passage分布")
    tf.logging.warn(answer_passage_dict)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title("question length dist")
    sns.distplot(question_len_list)
    plt.subplot(222)
    plt.title("passage length dist")
    sns.distplot(passage_len_list)
    plt.subplot(223)
    plt.title("passage num dist")
    sns.distplot(passage_num_list)
    if len(answer_len_list) > 0:
        plt.subplot(224)
        plt.title("answer length dist")
        sns.distplot((answer_len_list))
    # plt.savefig(os.path.split(path)[1].replace(".", "_"))
    plt.show()



def sample_data_to_read(path):
    tf.logging.warn(path)

    param = get_default_param()

    i = 0
    for sample in tqdm(load_dataset(os.path.join(ROOT_PATH, path), param)):
        if i == 0:
            tf.logging.info(sample.keys())

        tf.logging.info("question type: ")
        tf.logging.info(sample["question_type"])
        tf.logging.info("question tokens: ")
        tf.logging.info(sample["question_tokens"])

        # tf.logging.info("passage: ")
        # tf.logging.info(sample["passages"])

        if "answers" in sample.keys():
            tf.logging.info("answer: ")
            tf.logging.info(sample["answers"])
            pass
        if "answer_spans" in sample.keys():
            tf.logging.info("answer span: ")
            tf.logging.info(sample["answer_spans"])
            # if len(sample['answer_spans']) != 0:
            #     answer_len_list.append(sample["answer_spans"][0][1] - sample["answer_spans"][0][0])
        if 'answer_passages' in sample.keys():
            tf.logging.info("answer passages: ")
            tf.logging.info(sample["answer_passages"])
            # for x in sample["answer_passages"]:
            #     answer_passage_dict[x] += 1
            print("".join(sample["documents"][sample["answer_docs"][0]]["passage_tokens"][
                    sample["answer_spans"][0][0]:sample["answer_spans"][0][1] + 1]))
        time.sleep(10)


if __name__ == "__main__":

    # tf.logging.set_verbosity("WARN")
    tf.logging.set_verbosity("INFO")
    # for path in [TRAIN_SEARCH_PATH, TRAIN_ZD_PATH, DEV_SEARCH_PATH, DEV_ZD_PATH, TEST_SEARCH_PATH, TEST_ZD_PATH]:
    for path in [DEV_SEARCH_PATH, DEV_ZD_PATH, TEST_SEARCH_PATH, TEST_ZD_PATH]:
    #     statistic_plot(path)
        sample_data_to_read(path)