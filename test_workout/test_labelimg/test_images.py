import os
import unittest

from workout.image import Image
from workout.labelimg.data import Data
from workout.labelimg.images import Images


class TestImages(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    dofus = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus'

    def setUp(self):
        self.data = Data.factory(source=self.dofus)

    def test_(self):
        assert isinstance(Images.instance, Images)
        assert Images.instance.name == 'images'
        assert os.path.isdir(Images.instance.source)
        assert os.path.isdir(Images.instance.path)

    def test_images(self):
        images = Images.instance.images
        assert isinstance(Images.instance.images, list)
        assert all(isinstance(i, Image) for i in images)

    def test_format(self):
        assert all(i.format.lower().endswith('.jpg') for i in Images.instance.images)

    def test_all(self):
        assert Images.instance.all
