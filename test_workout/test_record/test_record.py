import os
import unittest

from workout.labelimg.data import Data
from workout.record.record import Record


class TestRecord(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    fps = 10

    def setUp(self):
        self.data = Data.factory(source=self.overwatch)
        self.record = Record.factory(source=Data.instance.path)

    def test_(self):
        assert isinstance(Record.instance, Record)
        assert Record.instance.name == 'record'
        assert os.path.isdir(Record.instance.source)
        assert os.path.isdir(Record.instance.path)

    def test_record_video(self):
        Record.instance.record_video(output='test_output.avi', title='Overwatch', fps=self.fps)
        assert os.path.isfile(os.path.join(Record.instance.path, 'test_output.avi'))
        assert os.path.getsize(os.path.join(Record.instance.path, 'test_output.avi')) > 0
