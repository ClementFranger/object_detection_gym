from test_workout import TestTensorflow
from workout.labelimg.data import Data
from workout.model.model import Model


class TestModel(TestTensorflow):
    num_classes, batch_size, num_steps = 6, 32, 10000

    def setUp(self):
        super().setUp()
        self.data = Data.factory(source=self.data)
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

