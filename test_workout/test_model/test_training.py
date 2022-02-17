import os

from test_workout import TestTensorflow
from workout.model.training import Training


class TestTraining(TestTensorflow):

    def setUp(self):
        super().setUp()
        self.training = Training.factory(source=self.model)

    def test_(self):
        assert isinstance(Training.instance, Training)
        assert Training.instance.name == 'training'
        assert os.path.isdir(Training.instance.path)
