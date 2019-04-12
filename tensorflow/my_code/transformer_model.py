#!/usr/bin/env python3
#!-*-coding=utf-8-*-
import time
import numpy as np
import tensorflow as tf

from my_code import transformer_utils
from my_code.transformer_utils import HParams
from rc_model import RCModel


class TransformerModel(RCModel):
    def __init__(self, vocab, args):

        # basic config
        self.n_head = 12
        self.n_layer = 10

        super(TransformerModel, self).__init__(vocab, args)

    def _build_graph(self):
        start_t = time.time()
        self._setup_placeholders()
        # self._embed()
        self._transformer()
        # self._encode()
        # self._match()
        # self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _transformer(self):
        hParam = HParams(
            n_vocab=self.vocab.size(),
            n_ctx=self.max_p_len,
            n_embd=self.hidden_size*2,
            n_head=12,
            n_layer=5,
        )
        self.fuse_p_encodes = transformer_utils.model(hParam, self.p, embeddings=self.vocab.embeddings)
        self.sep_q_encodes = transformer_utils.model(hParam, self.q, embeddings=self.vocab.embeddings, reuse=True)