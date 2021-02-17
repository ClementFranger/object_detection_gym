import os
import logging
from pathlib import Path

from workout.labelimg.labelimg import Data
from workout.labelimg.tfwriter import ConfigWriter

logger = logging.getLogger(__name__)


class Model:
    instance = None

    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        logger.info('Creating model factory from {path}'.format(path=self.path))
        PipelineConfig.factory(config=Path(os.path.join(kwargs.get('path'), kwargs.get('config', PipelineConfig.name))),
                               **kwargs)
        self.python = kwargs.get('python') or os.getenv('PYTHON')
        self.training = os.path.join(self.path, 'training')
        self.graph = os.path.join(self.path, 'graph')

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def config(self):
        return PipelineConfig.instance

    def run(self, **kwargs):
        main_train = kwargs.get('main_train') or os.getenv('MAIN_TRAIN')
        cmd = '{python} {main_train} --pipeline_config_path={config} --model_dir={training}'.format(
            python=self.python, main_train=main_train, config=self.config.config, training=self.training)
        print('Running command {cmd}'.format(cmd=cmd))
        os.system(cmd)

    def tensorboard(self, **kwargs):
        cmd = '{python} -m tensorboard.main --logdir={training}'.format(python=self.python, training=self.training)
        print('Running command {cmd}'.format(cmd=cmd))
        os.system(cmd)

    def save(self, **kwargs):
        main_graph = kwargs.get('main_graph') or os.getenv('MAIN_GRAPH')
        cmd = '{python} {main_graph} --input_type image_tensor --pipeline_config_path {config} --trained_checkpoint_dir {training} --output_directory {graph}'.format(
            python=self.python, main_graph=main_graph, config=self.config.config, training=self.training, graph=self.graph)
        print('Running command {cmd}'.format(cmd=cmd))
        os.system(cmd)


class PipelineConfig:
    instance = None
    name = 'pipeline.config'

    def __init__(self, **kwargs):
        self.config = kwargs.get('config')
        TrainConfig.factory(**kwargs)
        TrainInput.factory(**kwargs)
        TestInput.factory(**kwargs)
        SSD.factory(**kwargs)

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def writer(self):
        ConfigWriter.factory(path=self.config)
        return ConfigWriter.instance

    def update(self):
        read = self.writer.read()
        edit = self.writer.edit(pipeline=read, num_classes=SSD.instance.num_classes,
                                batch_size=TrainConfig.instance.batch_size, num_steps=TrainConfig.instance.num_steps,
                                label_map_path=TrainInput.instance.labels or TestInput.instance.labels,
                                train_input_path=TrainInput.instance.input, test_input_path=TestInput.instance.input,
                                fine_tune_checkpoint=TrainConfig.instance.fine_tune_checkpoint)
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
        # TODO : parse label_map.pbtxt to get num_classes


class TrainConfig(Config):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 512)
        self.num_steps = kwargs.get('num_steps', 20000)
        self.fine_tune_checkpoint = kwargs.get('fine_tune_checkpoint')


class Input(Config):
    name = None

    def __init__(self, **kwargs):
        self.data = kwargs.get('data') or Data.instance

    @property
    def labels(self):
        return os.path.join(self.data.path, 'labels.pbtxt')

    @property
    def input(self):
        return os.path.join(self.data.path, '{name}.record'.format(name=self.name))


class TrainInput(Input):
    name = 'train'


class TestInput(Input):
    name = 'test'
