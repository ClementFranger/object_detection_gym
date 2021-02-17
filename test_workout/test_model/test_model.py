import os
import unittest

from workout.labelimg.labelimg import LabelIMG, Data
from workout.model.model import Model, Config


class TestModel(unittest.TestCase):
    path = r'/workout/overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.model = Model.factory(path=self.path)

    def test_config(self):
        assert os.path.isfile(Config.instance.path)


class TestConfig(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'
    fine_tune_checkpoint = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\checkpoint\ckpt-0'
    num_classes = 1
    batch_size = 32
    num_steps = 10000

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.overwatch)
        self.model = Model.factory(path=self.path, num_classes=self.num_classes, batch_size=self.batch_size,
                                   num_steps=self.num_steps, data=Data.instance,
                                   fine_tune_checkpoint=self.fine_tune_checkpoint)

    def test_update(self):
        Model.instance.config.update()
        assert os.path.isfile(Model.instance.config.config)

    def test_run(self):
        Model.instance.run()

    def test_tensorboard(self):
        Model.instance.tensorboard()
