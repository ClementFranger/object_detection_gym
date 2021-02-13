import tensorflow as tf


class TFRecord:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        self.writer = tf.io.TFRecordWriter(self.path)
