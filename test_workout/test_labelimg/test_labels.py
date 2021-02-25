import os
import unittest

from workout.labelimg.data import Data
from workout.labelimg.labels import Labels


class TestLabels(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.data = Data.factory(source=self.overwatch)

    def test_(self):
        assert isinstance(Labels.instance, Labels)
        assert Labels.instance.name == 'labels'
        assert os.path.isdir(Labels.instance.source)
        assert os.path.isdir(Labels.instance.path)

    def test_split(self):
        assert True
