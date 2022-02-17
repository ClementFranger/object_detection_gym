import os

from test_workout import TestTensorflow
from workout.model.checkpoint import Checkpoint


class TestCheckpoint(TestTensorflow):

    def setUp(self):
        self.checkpoint = Checkpoint.factory(source=self.model)

    def test_(self):
        assert isinstance(Checkpoint.instance, Checkpoint)
        assert Checkpoint.instance.name == 'checkpoint'
        assert os.path.isdir(Checkpoint.instance.path)

    def test_checkpoint(self):
        checkpoint = Checkpoint.instance.checkpoint