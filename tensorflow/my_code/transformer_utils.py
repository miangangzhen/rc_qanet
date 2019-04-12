import tensorflow as tf
import numpy as np


class HParams(object):
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer):
        self.n_vocab = n_vocab
        # position embedding length
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer


def default_hparams():
    return HParams(
        n_vocab=100,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )


def shape_list(x):
    """
    Get tensor's shape, supporting static shape and dynamic shape
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def positions_for(tokens, past_length):
    """
    Combine past_length and sequence_length, then tile them to [batch_size, past_length + sequence_length]
    :tokens : input X.
    :past_length : ???.
    """
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    # tile None + range(sequence_length) to batch_size
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def expand_tile(value, size):
    """
    Add a new axis of given size

    """
    value = tf.convert_to_tensor(value, name="value")
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)


def block(x, scope, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        # multi head attention
        a, present = attn(norm(x, "ln_1"), "attn", nx, past=past, hparams=hparams)
        # residual
        x = x + a
        # linear projection and normalization
        m = mlp(norm(x, "ln_2"), "mlp", nx*4, hparams=hparams)
        # residual
        x = x + m
        return x, present


def mlp(x, scope, n_state, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, "c_fc", n_state))
        h2 = conv1d(h, "c_proj", nx)
        return h2


def norm(x, scope, axis=-1, epsilon=1e-5):
    """
    Apply norm function to x.
    mean_x, u = mean(x, axis=-1)
    var_x, s = mean_square(x - u)
    x = (x - mean_x) * 1 / (var_x + epsilong)
    linear projection, x = x * g + b
    """
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        # tf.multiply(x, g) + b
        x = x * g + b
        return x


def split_states(x, n):
    """
    Multi-head: reshape the last dimension of x into [n, x.shape[-1]/n]
    """
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n,  m//n])


def merge_states(x):
    """
    Merge the last two dimensions.
    """
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])


def attention_mask(nd, ns, dtype):
    """
    1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    example: attention_mask(5, 3, tf.int32)
    >>>
    [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ]
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def conv1d(x, scope, nf, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable("w", [nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), w) + b, start+[nf])
        return c


def attn(x, scope, n_state, past, hparams):
    """
    Attention function
    """
    # [batch_size, sequence_length, hidden_size]
    assert x.shape.ndims == 3
    assert n_state % hparams.n_head == 0
    if past is not None:
        # [batch_size, 2, heads, sequence_length, hidden_size], 2 is [k, v]
        assert past.shape.ndims == 5

    def split_heads(x):
        # [batch_size, sequence_length, hidden_size] => [batch_size, n_head, sequence_length, hidden_size//n_head]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch_size, n_head, dst_seq_len, src_seq_len]
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # As same as paper
        # q, k, v have shape [batch_size, n_head, sequence_length, hidden_size//n_head]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        # linear projection from x to k,q,v
        c = conv1d(x, "c_attn", n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state)
        return a, present


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def model(hparams, X, embeddings=None, past=None, scope="model", reuse=False):
    """
    :hparams: hyper parameters.
    :X : input X, shape = [batch_size, sequence_length].
    :past : ???.
    :scope : tensorflow graph scope name.
    :reuse : whether reuse variables or not.
    """
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        # position embedding
        wpe = tf.get_variable("wpe", [hparams.n_ctx, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.01))

        # word embedding
        pad = tf.get_variable("pad", shape=[1, hparams.n_embd], initializer=tf.zeros_initializer(), trainable=False)
        unk = tf.get_variable("unk", shape=[1, hparams.n_embd], initializer=tf.random_normal_initializer(stddev=0.01), trainable=True)
        w2v = tf.get_variable(
                'w2v',
                shape=(hparams.n_vocab, hparams.n_embd),
                initializer=tf.constant_initializer(embeddings),
                trainable=False
            )
        wte = tf.concat([pad, unk, w2v[2:]], axis=0)
        # wte = tf.get_variable("wte", [hparams.n_vocab, hparams.n_embd],
        #     initializer=tf.random_normal_initializer(stddev=0.02))

        # ???
        past_length = 0 if past is None else tf.shape(past)[-2]

        # embedding_lookup for word_embedding and position embedding
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # transformer
        # presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer

        for layer, past in enumerate(pasts):
            # h, present = block(h, "h{}".format(layer), past=past, hparams=hparams)
            h, _ = block(h, "h{}".format(layer), past=past, hparams=hparams)
            # presents.append(present)

        # results["present"] = tf.stack(presents, axis=1)
        h = norm(h, "ln_f")
        return tf.reshape(h, [batch, sequence, hparams.n_embd])

        # language model loss, do tokens < n predict token n?
        # h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        # # project h_flat to [batch_size * sequence_length, vocab_size]
        # logits = tf.matmul(h_flat, wte, transpose_b=True)
        # logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        # results["logits"] = logits
        # return results


if __name__ == "__main__":
    x = tf.constant(1, shape=[8, 20])
    result = model(default_hparams(), x)
    print(result)