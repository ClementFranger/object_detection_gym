import os
import unittest

from workout.labelimg.data import Data
from workout.labelimg.tfrecord import TFRecord
from workout.labelimg.train import Train


class TestTrain(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    dofus = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus'

    def setUp(self):
        self.data = Data.factory(source=self.dofus)

    def test_(self):
        assert isinstance(Train.instance, Train)
        assert Train.instance.name == 'train'
        assert os.path.isdir(Train.instance.source)
        assert os.path.isdir(Train.instance.path)

    def test_tfrecords(self):
        tfrecords = Train.instance.tfrecords
        assert isinstance(tfrecords, list)
        assert all(isinstance(t, TFRecord) for t in tfrecords)

    def test_tfrecords_write(self):
        Train.instance.tfrecords_write(path=Data.instance.train_records)
        assert os.path.isfile(Data.instance.train_records)
        assert os.path.getsize(Data.instance.train_records) > 0
