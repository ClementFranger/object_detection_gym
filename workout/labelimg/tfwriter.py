import os
import logging
import tensorflow as tf
from pathlib import Path

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

logger = logging.getLogger(__name__)


class TFWriter:
    instance = None

    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        logger.info('Initializing TFWriter to {path}'.format(path=self.path))
        self.writer = tf.io.TFRecordWriter(str(self.path))

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    def write(self, **kwargs):
        logger.info('Writing tfrecord to {path}'.format(path=self.path))
        self.writer.write(
            tf.train.Example(features=tf.train.Features(feature=kwargs.get('tfrecord'))).SerializeToString())


class TrainTFWriter(TFWriter):
    pass


class TestTFWriter(TFWriter):
    pass


class ConfigWriter:
    instance = None

    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        logger.info('Initializing ConfigWriter to {path}'.format(path=self.path))

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    def read(self):
        logger.info('Reading config from {path}'.format(path=self.path))
        pipeline = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(str(self.path), 'r') as f:
            text_format.Merge(f.read(), pipeline)
        return pipeline

    def edit(self, **kwargs):
        pipeline = kwargs.get('pipeline')

        logger.info('Editing num_classes to {num_classes}'.format(**kwargs))
        pipeline.model.ssd.num_classes = kwargs.get('num_classes')
        pipeline.train_config.fine_tune_checkpoint_type = 'detection'

        logger.info('Editing batch_size to {batch_size}'.format(**kwargs))
        pipeline.train_config.batch_size = kwargs.get('batch_size')
        logger.info('Editing num_steps to {num_steps}'.format(**kwargs))
        pipeline.train_config.num_steps = kwargs.get('num_steps')
        logger.info('Editing fine_tune_checkpoint to {fine_tune_checkpoint}'.format(**kwargs))
        pipeline.train_config.fine_tune_checkpoint = kwargs.get('fine_tune_checkpoint')

        logger.info('Editing label_map_path to {label_map_path}'.format(**kwargs))
        pipeline.train_input_reader.label_map_path = kwargs.get('label_map_path')
        pipeline.eval_input_reader[0].label_map_path = kwargs.get('label_map_path')

        logger.info('Editing train input_path to {train_input_path}'.format(**kwargs))
        pipeline.train_input_reader.tf_record_input_reader.input_path[0] = kwargs.get('train_input_path')

        logger.info('Editing test input_path to {test_input_path}'.format(**kwargs))
        pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = kwargs.get('test_input_path')

        return pipeline

    def write(self, **kwargs):
        logger.info('Writing config to {path}'.format(path=self.path))
        with tf.io.gfile.GFile(str(self.path), 'wb') as f:
            f.write(text_format.MessageToString(kwargs.get('pipeline')))
