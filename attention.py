import tensorflow as tf
from tensorflow.keras import layers

class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1))
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)
