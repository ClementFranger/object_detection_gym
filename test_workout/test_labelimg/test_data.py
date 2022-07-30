import os

from test_workout import TestTensorflow
from workout.labelimg.data import Data
from workout.labelimg.images import Images
from workout.labelimg.labels import Labels
from workout.labelimg.test import Test
from workout.labelimg.train import Train


class TestData(TestTensorflow):

    def setUp(self):
        super().setUp()
        self.data = Data.factory(source=self.data)

    def test_(self):
        assert isinstance(Data.instance, Data)
        assert Data.instance.name == 'data'
        assert os.path.isdir(Data.instance.source)
        assert os.path.isdir(Data.instance.path)

    def test_images(self):
        assert isinstance(Data.instance.images, Images)
        assert os.path.isdir(Data.instance.images.path)

    def test_labels(self):
        assert isinstance(Data.instance.labels, Labels)
        assert os.path.isdir(Data.instance.labels.path)

    def test_train(self):
        assert isinstance(Data.instance.train, Train)
        assert os.path.isdir(Data.instance.train.path)

    def test_test(self):
        assert isinstance(Data.instance.test, Test)
        assert os.path.isdir(Data.instance.test.path)

    def test_labels_pbtxt(self):
        assert os.path.isfile(Data.instance.labels_pbtxt)

    def test_train_records(self):
        assert os.path.isfile(Data.instance.train_records)

    def test_test_records(self):
        assert os.path.isfile(Data.instance.test_records)

    def test_tfrecords_write(self):
        Data.instance.tfrecords_write()
        assert os.path.isfile(Data.instance.train_records)
        assert os.path.getsize(Data.instance.train_records) > 0
        assert os.path.isfile(Data.instance.test_records)
        assert os.path.getsize(Data.instance.test_records) > 0