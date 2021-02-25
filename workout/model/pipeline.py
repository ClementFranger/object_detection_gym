import os
import logging

from workout.labelimg.data import Data
from workout.model.checkpoint import Checkpoint
from workout.utils import Source

logger = logging.getLogger(__name__)


class PipelineConfig(Source):
    """ file containing pipeline config """
    name = 'pipeline.config'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        TrainConfig.factory(**kwargs)
        TrainInput.factory(**kwargs)
        TestInput.factory(**kwargs)
        SSD.factory(**kwargs)

    @property
    def train_config(self):
        return TrainConfig.instance

    @property
    def train_input(self):
        return TrainInput.instance

    @property
    def test_input(self):
        return TestInput.instance

    @property
    def ssd(self):
        return SSD.instance

    @property
    def writer(self):
        ConfigWriter.factory(path=self.path)
        return ConfigWriter.instance

    def update(self):
        logger.info('Updating pipeline config')
        read = self.writer.read()
        edit = self.writer.edit(pipeline=read, num_classes=self.ssd.num_classes,
                                batch_size=self.train_config.batch_size, num_steps=self.train_config.num_steps,
                                label_map_path=self.train_input.labels or self.test_input.labels,
                                train_input_path=self.train_input.input, test_input_path=self.test_input.input,
                                fine_tune_checkpoint=self.train_config.fine_tune_checkpoint)
        write = self.writer.write(pipeline=edit)


class Config:
    instance = None

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance


class SSD(Config):
    def __init__(self, **kwargs):
        self.num_classes = kwargs.get('num_classes', 1)
        # TODO : parse label_map.pbtxt to get num_classes using label_map_util


class TrainConfig(Config):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 512)
        self.num_steps = kwargs.get('num_steps', 20000)
        self.fine_tune_checkpoint = kwargs.get('fine_tune_checkpoint') or Checkpoint.instance.checkpoint


class Input(Config):
    name = None

    def __init__(self, **kwargs):
        self.data = kwargs.get('data') or Data.instance.path
        assert os.path.isdir(self.data)

    @property
    def labels(self):
        return os.path.join(self.data, 'labels.pbtxt')

    @property
    def input(self):
        return os.path.join(self.data, '{name}.record'.format(name=self.name))


class TrainInput(Input):
    name = 'train'


class TestInput(Input):
    name = 'test'
