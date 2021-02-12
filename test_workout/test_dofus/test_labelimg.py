import os
import unittest

from workout.dofus.labelimg import DofusLabelIMG


class TestLabelIMG(unittest.TestCase):

    def setUp(self):
        self.dofus = DofusLabelIMG.factory()

    def test_data(self):
        data = DofusLabelIMG.instance.data

        assert os.path.isdir(data)

    def test_images(self):
        images = DofusLabelIMG.instance.images

        assert os.path.isdir(images)

    def test_labels(self):
        labels = DofusLabelIMG.instance.labels

        assert os.path.isdir(labels)

    def test_train(self):
        train = DofusLabelIMG.instance.train

        assert os.path.isdir(train)

    def test_test(self):
        test = DofusLabelIMG.instance.test

        assert os.path.isdir(test)
