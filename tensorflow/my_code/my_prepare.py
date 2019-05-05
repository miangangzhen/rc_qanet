import logging
import tensorflow as tf
from dataset import BRCDataset
import collections
from vocab import Vocab
import os
import pickle
import json
from collections import Counter

"""
用这个程序替代官方run.py里的prepare方法。
准备vocab文件，保存预训练的词向量
"""


class Param(collections.namedtuple("param",
                                   ["train_files","dev_files", "test_files",
                                    "vocab_dir", "model_dir", "result_dir", "summary_dir",
                                    "max_p_num", "max_p_len", "max_q_len",
                                    "embed_size", "use_embd"])):
    pass


# override /DuReader/tensorflow/dataset.BRCDataset._load_dataset + word_iter
def load_dataset(path, param, train=False):
    for data_path in tf.gfile.Glob(path):
        with open(data_path, "r", encoding="utf-8") as fin:
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= param.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens, p_tokens, title, title_tokens in zip(doc['segmented_paragraphs'],
                                                                              doc['paragraphs'], doc['title'],
                                                                              doc['segmented_title']):
                            question_tokens = sample['segmented_question']
                            p_char_tokens = [i for i in p_tokens]
                            title_char_tokens = [i for i in title]
                            q_char_tokens = [i for i in sample['question']]

                            # 这里改过了，和原版是不一样的，计算recall_wrt_question分数的方法被修改了
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            common_with_question_char = Counter(p_char_tokens) & Counter(q_char_tokens)
                            title_common_with_question = Counter(title_tokens) & Counter(question_tokens)
                            title_common_with_question_char = Counter(title_char_tokens) & Counter(q_char_tokens)

                            correct_preds = sum(common_with_question.values())
                            recall_eq = 0 if correct_preds == 0 else float(correct_preds) / len(question_tokens)

                            correct_preds = sum(common_with_question_char.values())
                            recall_eq_char = 0 if correct_preds == 0 else float(correct_preds) / len(q_char_tokens)

                            correct_preds = sum(title_common_with_question.values())
                            recall_tq = 0 if correct_preds == 0 else float(correct_preds) / len(question_tokens)

                            correct_preds = sum(title_common_with_question_char.values())
                            recall_tq_char = 0 if correct_preds == 0 else float(correct_preds) / len(q_char_tokens)

                            recall_wrt_question = float(recall_eq + recall_eq_char) / 2
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                yield sample


# override /DuReader/tensorflow/run.prepare
def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in tf.gfile.Glob(args.train_files + args.dev_files + args.test_files):
        assert tf.gfile.Exists(data_path), '{} file does not exist.'.format(data_path)

    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = load_dataset(param.train_files, param, train=True)
    vocab = Vocab(lower=True)
    for sample in brc_data:
        for token in sample['question_tokens']:
            vocab.add(token)
        for passage in sample['passages']:
            for token in passage['passage_tokens']:
                vocab.add(token)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=10)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    if args.use_embd:
        vocab.load_pretrained_embeddings("100_ver_not_pure.txt")
    else:
        vocab.randomly_init_embeddings(args.embed_size)

    logging.info(vocab)
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("brc")

    working_dir = "../../data/"
    param = Param(["../../../../data/train_preprocessed/trainset/*"],
                  dev_files=[],
                  test_files=[],
                  vocab_dir=working_dir+"vocab/",
                  model_dir=working_dir+"models/",
                  result_dir=working_dir+"results/",
                  summary_dir=working_dir+"summary/",
                  max_p_num=5,
                  max_p_len=500,
                  max_q_len=60,
                  embed_size=100,
                  use_embd=True)

    prepare(param)
