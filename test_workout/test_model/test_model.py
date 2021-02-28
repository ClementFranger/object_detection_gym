import unittest

from workout.labelimg.data import Data
from workout.model.model import Model


class TestModel(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'
    num_classes, batch_size, num_steps = 1, 32, 10000

    def setUp(self):
        self.data = Data.factory(source=self.overwatch)
        self.model = Model.factory(path=self.model, num_classes=self.num_classes,
                                   batch_size=self.batch_size, num_steps=self.num_steps)

    def test_update(self):
        Model.instance.update()

    def test_train(self):
        Model.instance.train()

    def test_tensorboard(self):
        Model.instance.tensorboard()

    def test_save(self):
        Model.instance.save()

