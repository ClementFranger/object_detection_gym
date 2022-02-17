import os

from test_workout import TestTensorflow
from workout.labelimg.data import Data
from workout.labelimg.labels import Labels


class TestLabels(TestTensorflow):

    def setUp(self):
        super().setUp()
        self.data = Data.factory(source=self.data)

    def test_(self):
        assert isinstance(Labels.instance, Labels)
        assert Labels.instance.name == 'labels'
        assert os.path.isdir(Labels.instance.source)
        assert os.path.isdir(Labels.instance.path)

    def test_split(self):
        assert True
