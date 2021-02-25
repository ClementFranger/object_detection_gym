import os
import logging
import tensorflow as tf
from pathlib import Path

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

logger = logging.getLogger(__name__)


# class TFRecordWriter:
#     def __init__(self, **kwargs):
#         self.path = Path(kwargs.get('path'))
#         logger.info('Initializing TFRecordWriter to {path}'.format(path=self.path))
#         self.writer = tf.io.TFRecordWriter(str(self.path))
#
#     def write(self, **kwargs):
#         self.writer.write(
#             tf.train.Example(features=tf.train.Features(feature=kwargs.get('tfrecord'))).SerializeToString())
#
#
# class TFWriter:
#     instance = None
#
#     def __init__(self, **kwargs):
#         self.path = Path(kwargs.get('path'))
#         logger.info('Initializing TFWriter to {path}'.format(path=self.path))
#         self.writer = tf.io.TFRecordWriter(str(self.path))
#
#     @classmethod
#     def factory(cls, **kwargs):
#         if cls.instance is None:
#             cls.instance = cls(**kwargs)
#         assert isinstance(cls.instance, cls)
#         return cls.instance
#
#     def write(self, **kwargs):
#         logger.info('Writing tfrecord to {path}'.format(path=self.path))
#         self.writer.write(
#             tf.train.Example(features=tf.train.Features(feature=kwargs.get('tfrecord'))).SerializeToString())


# class TrainTFWriter(TFWriter):
#     pass
#
#
# class TestTFWriter(TFWriter):
#     pass



