import os
import unittest

from workout.labelimg.data import Data
from workout.labelimg.tfrecord import TFRecord
from workout.labelimg.test import Test


class TestTest(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.data = Data.factory(source=self.overwatch)

    def test_(self):
        assert isinstance(Test.instance, Test)
        assert Test.instance.name == 'test'
        assert os.path.isdir(Test.instance.source)

    def test_tfrecords(self):
        tfrecords = Test.instance.tfrecords
        assert isinstance(tfrecords, list)
        assert all(isinstance(t, TFRecord) for t in tfrecords)

    def test_tfrecords_write(self):
        Test.instance.tfrecords_write(path=Data.instance.test_records)
        assert os.path.isfile(Data.instance.train_records)
        assert os.path.getsize(Data.instance.train_records) > 0
