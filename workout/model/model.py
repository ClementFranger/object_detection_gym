import os
import logging
import tensorflow as tf

from workout.labelimg.labelimg import Data
from workout.labelimg.tfwriter import ConfigWriter
from workout.utils import Source

logger = logging.getLogger(__name__)


class Model(Source):
    instance = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        PipelineConfig.factory(source=self.path)
        Checkpoint.factory(source=self.path)
        Training.factory(source=self.path)
        Graph.factory(source=self.path)
        self.python = kwargs.get('python') or os.getenv('PYTHON')

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def pipeline_config(self):
        return PipelineConfig.instance

    @property
    def checkpoint(self):
        return Checkpoint.instance

    @property
    def training(self):
        return Training.instance

    @property
    def graph(self):
        return Graph.instance

    def train(self, **kwargs):
        main_train = kwargs.get('main_train') or os.getenv('MAIN_TRAIN')
        assert os.path.isfile(self.pipeline_config.train_input.input)
        assert os.path.isfile(self.pipeline_config.test_input.input)
        assert os.path.isfile(self.pipeline_config.train_input.labels)
        cmd = '{python} {main_train} --pipeline_config_path={config} --model_dir={training}'.format(
            python=self.python, main_train=main_train, config=self.pipeline_config.path, training=self.training.path)
        print('Running command {cmd}'.format(cmd=cmd))
        os.system(cmd)

    def tensorboard(self, **kwargs):
        cmd = '{python} -m tensorboard.main --logdir={training}'.format(python=self.python, training=self.training.path)
        print('Running command {cmd}'.format(cmd=cmd))
        os.system(cmd)

    def save(self, **kwargs):
        main_graph = kwargs.get('main_graph') or os.getenv('MAIN_GRAPH')
        cmd = '{python} {main_graph} --input_type image_tensor --pipeline_config_path {config} --trained_checkpoint_dir {training} --output_directory {graph}'.format(
            python=self.python, main_graph=main_graph, config=self.pipeline_config.path, training=self.training.path, graph=self.graph.path)
        print('Running command {cmd}'.format(cmd=cmd))
        os.system(cmd)


class Checkpoint(Source):
    name = 'checkpoint'

    @property
    def checkpoint(self):
        return os.path.join(self.path, 'ckpt-0')


class Training(Source):
    name = 'training'


class Graph(Source):
    name = 'graph'

    @property
    def saved_model(self):
        return os.path.join(self.path, 'saved_model')

    @property
    def graph(self):
        return tf.saved_model.load(self.saved_model, tags=None, options=None)


class PipelineConfig(Source):
    instance = None
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
