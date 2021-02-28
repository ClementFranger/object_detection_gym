import os
import unittest

from workout.model.training import Training


class TestTraining(unittest.TestCase):
    model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.training = Training.factory(source=self.model)

    def test_(self):
        assert isinstance(Training.instance, Training)
        assert Training.instance.name == 'training'
        assert os.path.isdir(Training.instance.path)
