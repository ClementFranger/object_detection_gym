import os
import unittest

from tensorflow.python.saved_model.load import _WrapperFunction

from workout.detection.detection import Detection
from workout.detection.model import Model


class TestModel(unittest.TestCase):
    graph = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph'

    def setUp(self):
        self.detection = Detection.factory(source=self.graph)

    def test_load(self):
        assert isinstance(Model.instance.model, _WrapperFunction)