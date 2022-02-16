import os

from test_workout import TestTensorflow
from workout.model.graph import Graph


class TestGraph(TestTensorflow):

    def setUp(self):
        self.graph = Graph.factory(source=self.dofus_model)

    def test_(self):
        assert isinstance(Graph.instance, Graph)
        assert Graph.instance.name == 'graph'
        assert os.path.isdir(Graph.instance.path)

    def test_saved_model(self):
        assert os.path.isdir(Graph.instance.saved_model)
