import unittest

from pyrect import Size

from workout.record.application import Application


class TestApplication(unittest.TestCase):

    def setUp(self):
        self.application = Application.factory(title='Overwatch')

    def test_(self):
        assert Application.instance._window

    def test_size(self):
        assert isinstance(Application.instance.size, Size)
        assert Application.instance.size == (1920, 1080)
