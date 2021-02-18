import os
import unittest

from workout.video.video import Video, Application


class TestVideo(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data'

    def setUp(self):
        self.video = Video.factory(source=self.path, title='Overwatch')

    def test_record(self):
        Video.instance.record()
        # assert os.path.isfile(Video.instance.path)