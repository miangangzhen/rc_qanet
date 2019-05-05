# -*- coding:utf8 -*-
from collections import Counter
from dataset import BRCDataset
import logging
import tensorflow as tf
import json
import random
"""
清洗训练数据，以替代train_preprocessed/trainset/目录下的训练数据
1.训练数据筛选
2.数据增强: 原始数据num_passage=5，1真4假。这里改成了num_p=2，1真1假
"""

class DataClean(BRCDataset):
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[], limit=None):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.alphabets_set = set([x for x in "abcdefghijklmnopqlstuvwxyz"])

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in tf.gfile.Glob(train_files):
                self._transform_dataset(train_file, train=True, limit=limit)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        #
        # if dev_files:
        #     for dev_file in tf.gfile.Glob(dev_files):
        #         self.dev_set += self._load_dataset(dev_file)
        #     self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
        #
        # if test_files:
        #     for test_file in tf.gfile.Glob(test_files):
        #         self.test_set += self._load_dataset(test_file)
        #     self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _transform_dataset(self, data_path, train=False, limit=None):
        """
        训练预料处理
        1.训练数据筛选: answer与answer_passage[start, end]重合率低的数据被滤掉。
        2.数据增强: num_passage=5 1真4假 改为 num_passage=2 1真1假 * 4
        Args:
            data_path: the data file to load
        """
        skip_count = 0
        with open(data_path + "cleaned", "w", encoding="utf-8") as fout:
            with open(data_path, encoding="utf-8") as fin:
                for lidx, line in enumerate(fin):
                    # 照抄源码，解析数据 ###################################################
                    if limit is not None and lidx > limit:
                        break
                    data = {}
                    sample = json.loads(line.strip())
                    if train:
                        if len(sample['answer_spans']) == 0:
                            skip_count += 1
                            continue

                    # if "answer_spans" in sample.keys():
                    data["answer_spans"] = sample["answer_spans"]
                    # if 'answer_docs' in sample:
                    data['answer_docs'] = sample['answer_docs']
                    if sample["answer_docs"][0] > (len(sample["documents"])-1):
                        print("answer doc not in documents")
                        print(sample)
                        continue

                    data['segmented_question'] = sample['segmented_question']
                    data["question_id"] = sample["question_id"]
                    data["question_type"] = sample["question_type"]
                    if "answers" in sample.keys():
                        data["answers"] = sample["answers"]

                    data['documents'] = []
                    for d_idx, doc in enumerate(sample['documents']):
                        if train:
                            # 训练模式，直接找到标注的最优doc-para
                            most_related_para = doc['most_related_para']
                            data['documents'].append(
                                {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                                 'is_selected': doc['is_selected']}
                            )
                    ############################################################

                    # 训练数据筛选 #############################################
                    # 检查人工编写的answer与span_start,span_end截取的内容是否相关
                    answer_length = len(data["answers"])
                    if answer_length > 0:
                        best_score = 0.0

                        selected_answer_set = set("".join(data["documents"][data["answer_docs"][0]]["passage_tokens"][
                                                  data["answer_spans"][0][0]:data["answer_spans"][0][1] + 1]))
                        for i in range(answer_length):
                            answer_set = set(data["answers"][i])
                            count = len(selected_answer_set & answer_set)

                            score = count / (len(answer_set) + 1)
                            if score > best_score:
                                best_score = score
                        if best_score < 0.01:
                            skip_count += 1
                            # print(data["answers"])
                            # print("".join(data["documents"][data["answer_docs"][0]]["passage_tokens"][
                            #                       data["answer_spans"][0][0]:data["answer_spans"][0][1] + 1]))
                            # print("====")
                            continue
                    ##########################################################

                    # 数据增强 ###############################################
                    # 删除训练中不需要的字段，减小训练文件大小
                    for doc in sample["documents"]:
                        doc.pop("paragraphs")
                        doc.pop("segmented_title")
                        doc.pop("title")

                    origin_answer_doc_id = sample['answer_docs'][0]
                    passage_num = len(sample["documents"])
                    origin_no_answer_doc_ids = [x for x in range(passage_num) if x != origin_answer_doc_id]
                    for id_ in origin_no_answer_doc_ids:
                        data["documents"] = []
                        if random.random() > 0.5:
                            data["documents"] = [sample["documents"][origin_answer_doc_id], sample["documents"][id_]]
                            data["answer_docs"] = [0]
                        else:
                            data["documents"] = [sample["documents"][id_], sample["documents"][origin_answer_doc_id]]
                            data["answer_docs"] = [1]
                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(skip_count)
        return None

if __name__ == "__main__":
    brc_data = DataClean(5, 500, 60,
                          # ["../data/train_preprocessed/trainset/search.train.json", "../data/train_preprocessed/trainset/zhidao.train.json"],
                         ["../data/train_preprocessed/trainset/search.train.json_pe"],
                         [], [], limit=None)