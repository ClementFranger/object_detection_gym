import os
import logging

from workout.model.checkpoint import Checkpoint
from workout.model.graph import Graph
from workout.model.pipeline import PipelineConfig
from workout.model.training import Training
from workout.utils import Source

logger = logging.getLogger(__name__)


class Model(Source):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs.pop('path')
        Checkpoint.factory(source=self.path)
        Training.factory(source=self.path)
        Graph.factory(source=self.path)
        PipelineConfig.factory(source=self.path, **kwargs)
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

    def update(self):
        self.pipeline_config.update()

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
