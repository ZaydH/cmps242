import tensorflow as tf

import const

def init(X, vocab):
  word_embeddings = tf.get_variable("word_embeddings", [len(vocab), const.EMBEDDING_RANK])
  vals = [v for (k, v) in vocab.items()]
  embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, vals)
  return tf.matmul(X, embedded_word_ids)
