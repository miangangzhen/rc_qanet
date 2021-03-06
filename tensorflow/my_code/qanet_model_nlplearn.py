import json
import os
import tensorflow as tf
from my_code.qanet_layers_nlplearn import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention
import logging
import time
from utils.dureader_eval import compute_bleu_rouge
from utils.dureader_eval import normalize


class QANET_Model_NLPLEARN(object):
    # def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo = False, graph = None):
    def __init__(self, vocab, config):

        # logging
        self.logger = logging.getLogger("brc")
        self.config = config

        # basic config
        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.use_dropout = config.dropout_keep_prob < 1

        # length limit
        self.max_p_num = config.max_p_num
        self.logger.info("numbers of passages %s" % self.max_p_num)


        self.max_p_len = config.max_p_len
        self.max_q_len = config.max_q_len
        self.max_a_len = config.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

        # checkpoint dir
        self.checkpoint_dir = config.checkpoint_dir


    def _build_graph(self):
        """
                Builds the computation graph with Tensorflow
                """
        start_t = time.time()
        self._create_model()

        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = total_params()
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _create_model(self):

        self.c = tf.placeholder(tf.int32, [None, None], "context")
        self.q = tf.placeholder(tf.int32, [None, None], "question")

        self.c = tf.reshape(self.c, [-1, self.config.max_p_len])
        self.q = tf.reshape(self.q, [-1, self.config.max_q_len])
        # self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit], "context_char")
        # self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit], "question_char")
        # self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index1")
        # self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index2")
        self.y1 = tf.placeholder(tf.int32, [None], "answer_label1")
        self.y2 = tf.placeholder(tf.int32, [None], "answer_label2")

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        #######################################################

        # self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
        #                                    initializer=tf.constant_initializer(0), trainable=False)
        # self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        # if self.demo:
        #     self.c = tf.placeholder(tf.int32, [None, config.test_para_limit],"context")
        #     self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit],"question")
        #     self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit],"context_char")
        #     self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit],"question_char")
        #     self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index1")
        #     self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index2")
        # else:
        #     self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()

        # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
        # self.word_mat = tf.get_variable("word_mat", [self.vocab.size(), self.vocab.embed_dim], initializer=xavier_initializer(), dtype=tf.float32, trainable=True)
        pretrained_word_mat = tf.get_variable("word_emb_mat",
                                                   [self.vocab.size() - 2, self.vocab.embed_dim],
                                                   dtype=tf.float32,
                                                   # initializer=xavier_initializer(),
                                                   initializer=tf.constant_initializer(
                                                       self.vocab.embeddings[2:],
                                                       dtype=tf.float32),
                                                   trainable=False)
        word_pad_unk_mat = tf.get_variable("word_unk_pad",
                                                [2, pretrained_word_mat.get_shape()[1]],
                                                dtype=tf.float32,
                                                initializer=tf.constant_initializer(
                                                    self.vocab.embeddings[:2],
                                                    dtype=tf.float32),
                                                trainable=True)

        self.word_mat = tf.concat([word_pad_unk_mat, pretrained_word_mat], axis=0)
        # self.char_mat = tf.get_variable(
        #     "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.forward()

        self.lr = tf.minimum(self.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, self.config.max_norm_grad)
        self.train_op = self.opt.apply_gradients(
            zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, d, nh = tf.shape(self.y1)[0], self.max_p_len, self.max_q_len, self.config.hidden_size, self.config.head_size

        with tf.variable_scope("Input_Embedding_Layer"):

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            q = residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.max_q_len,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.max_p_len,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.max_p_len, self.max_q_len, input_keep_prob =1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis = -1)
            self.enc = [conv(inputs, d, name = "input_projection")]
            for i in range(3):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.c_mask,
                        num_filters = d,
                        num_heads = nh,
                        seq_len = self.c_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout)
                    )
            for i, item in enumerate(self.enc):
                self.enc[i] = tf.reshape(self.enc[i],
                                         [N, -1, self.enc[i].get_shape()[-1]])

        with tf.variable_scope("Output_Layer"):
            start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
            end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
            reshape_mask = tf.reshape(self.c_mask, [N, -1])
            self.logits = [mask_logits(start_logits, mask = reshape_mask),
                           mask_logits(end_logits, mask = reshape_mask)]

            logits1, logits2 = [l for l in self.logits]

            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, config.max_a_len)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

            self.start_label_onehot = tf.one_hot(self.y1, tf.shape(logits1)[1], axis=1)
            self.end_label_onehot = tf.one_hot(self.y2, tf.shape(logits2)[1], axis=1)

            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.start_label_onehot)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.end_label_onehot)
            self.loss = tf.reduce_mean(losses + losses2)

        if config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var,v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def _train_epoch(self, train_batches, dropout):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.c: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         # self.qh: batch['question_char_ids'],
                         # self.ch: batch["passage_char_ids"],
                         self.y1: batch['start_id'],
                         self.y2: batch['end_id'],
                         self.dropout: dropout}

            # logits , y1 = self.sess.run([self.logits, self.y1], feed_dict=feed_dict)
            # exit(1)
            _, loss, global_step = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict)
            # _, loss, global_step, start_label, end_label, logit1, logit2 = self.sess.run([self.train_op, self.loss, self.global_step, self.start_label, self.end_label,self.logits1,self.logits2], feed_dict)
            # print(start_label)
            # print(logit1)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
                # save to checkpoint
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "save_every_log_every_n_batch"), global_step=global_step)
        print("total_num", total_num)
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=0.0, evaluate=True):
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_rouge_l = 0
        dropout = (1.*10 - dropout_keep_prob*10)/10
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches_for_qanet('train', batch_size, pad_id, shuffle=True)
            #  data.next_batch('train', batch_size, pad_id, pad_char_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches_for_qanet('dev', batch_size, pad_id, shuffle=False)
                    #  data.next_batch('dev', batch_size, pad_id, pad_char_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        self.save(save_dir, save_prefix)
                        max_rouge_l = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):

            feed_dict = {self.c: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.y1: batch['start_id'],
                         self.y2: batch['end_id'],
                         self.dropout: 0.0}

            try:
                start_probs, end_probs, loss = self.sess.run([self.logits1,
                                                              self.logits2, self.loss], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])

                padded_p_len = len(batch['passage_token_ids'][0])
                for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                    best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                    if save_full_info:
                        sample['pred_answers'] = [best_answer]
                        pred_answers.append(sample)
                    else:
                        pred_answers.append({'question_id': sample['question_id'],
                                             'question_type': sample['question_type'],
                                             'answers': [best_answer],
                                             'entity_answers': [[]],
                                             'yesno_answers': []})
                    if 'answers' in sample:
                        ref_answers.append({'question_id': sample['question_id'],
                                            'question_type': sample['question_type'],
                                            'answers': sample['answers'],
                                            'entity_answers': [[]],
                                            'yesno_answers': []})

            except:
                continue

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))