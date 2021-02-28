import os
import unittest

from workout.model.checkpoint import Checkpoint


class TestCheckpoint(unittest.TestCase):
    model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.checkpoint = Checkpoint.factory(source=self.model)

    def test_(self):
        assert isinstance(Checkpoint.instance, Checkpoint)
        assert Checkpoint.instance.name == 'checkpoint'
        assert os.path.isdir(Checkpoint.instance.path)

    def test_checkpoint(self):
        checkpoint = Checkpoint.instance.checkpoint