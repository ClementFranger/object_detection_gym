import os
import unittest

from workout.labelimg.data import Data
from workout.labelimg.record import Record


class TestVideo(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.data = Data.factory(source=self.overwatch)
        self.video = Record.factory(source=Data.instance.path)

    def test_(self):
        assert isinstance(Record.instance, Record)
        assert Record.instance.name == 'record'
        assert os.path.isdir(Record.instance.source)
        assert os.path.isdir(Record.instance.path)

    def test_record(self):
        Record.instance.record(title='Overwatch')
        # assert os.path.isfile(Video.instance.path)