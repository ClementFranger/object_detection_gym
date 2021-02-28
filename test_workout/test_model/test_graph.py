import os
import unittest

from workout.model.graph import Graph


class TestGraph(unittest.TestCase):
    model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.graph = Graph.factory(source=self.model)

    def test_(self):
        assert isinstance(Graph.instance, Graph)
        assert Graph.instance.name == 'graph'
        assert os.path.isdir(Graph.instance.path)

    def test_saved_model(self):
        assert os.path.isdir(Graph.instance.saved_model)
