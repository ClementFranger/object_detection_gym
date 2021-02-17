import os
import unittest

from workout.video.video import Video, Application


class TestVideo(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\test_workout\test_video\record'

    def setUp(self):
        self.video = Video.factory(path=self.path, title='Overwatch')

    def test_record(self):
        print(Application.instance.monitor)
        Video.instance.record()
        # assert os.path.isfile(Video.instance.path)