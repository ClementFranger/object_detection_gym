import logging
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)


class TFRecordWriter:
    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        logger.info('Initializing TFRecordWriter to {path}'.format(path=self.path))
        self.writer = tf.io.TFRecordWriter(str(self.path))

    def write(self, **kwargs):
        self.writer.write(
            tf.train.Example(features=tf.train.Features(feature=kwargs.get('tfrecord'))).SerializeToString())