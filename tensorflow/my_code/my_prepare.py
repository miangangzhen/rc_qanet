import logging
import tensorflow as tf
import collections
from vocab import Vocab
import os
import pickle
import json

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
def load_dataset(paths, param, train=False):
    for path in paths:
        for data_path in tf.gfile.Glob(path):
            print(data_path)
            with open(data_path, "r", encoding="utf-8") as fin:
                for lidx, line in enumerate(fin):
                    sample = json.loads(line.strip())
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
    for dir_path in [args.vocab_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = load_dataset([args.train_files, args.dev_files, args.test_files], args, train=True)
    vocab = Vocab(lower=True)
    for sample in brc_data:
        for token in sample['segmented_question']:
            vocab.add(token)
        for doc in sample['documents']:
            for p in doc['segmented_paragraphs']:
                for token in p:
                    vocab.add(token)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=10)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    if args.use_embd:
        vocab.load_pretrained_embeddings("w2v_pe.txt")
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

    working_dir = "./"
    # param = Param(["../data/train_preprocessed/trainset/*.json_pe"],
    #               dev_files=["../data/dev_preprocessed/devset/*.json_pe"],
    #               test_files=["../data/test1_preprocessed/test1set/*.json_pe"],
    #               vocab_dir=working_dir+"vocab/",
    #               model_dir=working_dir+"models/",
    #               result_dir=working_dir+"results/",
    #               summary_dir=working_dir+"summary/",
    #               max_p_num=5,
    #               max_p_len=800,
    #               max_q_len=60,
    #               embed_size=100,
    #               use_embd=True)
    #
    # prepare(param)

    with open("resource/vocab/vocab.data", "rb") as f:
        vocab = pickle.load(f)
    pass