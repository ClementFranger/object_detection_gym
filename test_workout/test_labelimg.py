import os
import unittest
import tensorflow as tf
from collections import namedtuple

from workout.labelimg import Train, Test, Data, Images, Labels, LabelIMG, XML, Root, Tree, Size, BNDBOX, TFRecord


class TestLabelIMG(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.path)

    def test_data(self):
        assert os.path.isdir(Data.instance.path)
        assert os.path.isdir(Images.instance.path)
        assert os.path.isdir(Labels.instance.path)
        assert os.path.isdir(Train.instance.path)
        assert os.path.isdir(Test.instance.path)


class TestTrain(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data'

    def setUp(self):
        self.data = Data.factory(path=self.path)

    def test_train_xml(self):
        xml = Train.instance.xml
        assert isinstance(xml, list)
        assert len(xml) == 4
        assert all(isinstance(x, XML) for x in xml)

    def test_train_tfrecord(self):
        tfrecord = Train.instance.tfrecord
        assert isinstance(tfrecord, list)
        assert len(tfrecord) == 4
        assert all(isinstance(x, TFRecord) for x in tfrecord)

    def test_train_write_tfrecord(self):
        """ do not run multiple times as it will append to current record """
        Train.instance.write_tfrecord()


class TestTest(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data'

    def setUp(self):
        self.data = Data.factory(path=self.path)

    def test_test_xml(self):
        xml = Test.instance.xml
        assert isinstance(xml, list)
        assert len(xml) == 1
        assert all(isinstance(x, XML) for x in xml)

    def test_test_tfrecord(self):
        tfrecord = Test.instance.tfrecord
        assert isinstance(tfrecord, list)
        assert len(tfrecord) == 1
        assert all(isinstance(x, TFRecord) for x in tfrecord)

    def test_test_write_tfrecord(self):
        """ do not run multiple times as it will append to current record """
        Test.instance.write_tfrecord()


class TestXML(unittest.TestCase):
    xml = r'C:\Users\Minifranger\Documents\python_scripts\workout\test_workout\mocks\test_xml.xml'

    def setUp(self):
        self.xml = XML(path=self.xml)

    def test_xml(self):
        assert os.path.isfile(self.xml.path)
        assert isinstance(self.xml, XML)
        assert isinstance(self.xml.tree, Tree)

    def test_root(self):
        root = self.xml.tree.root
        assert isinstance(root, Root)
        assert root.filename == '1.jpg'

    def test_size(self):
        size = self.xml.tree.root.size
        assert isinstance(size, Size)
        assert size.width == 296
        assert size.height == 171

    def test_object(self):
        objects = self.xml.tree.root.objects
        assert isinstance(objects, list)
        assert len(objects) > 0
        object = next(iter(objects))
        assert object.name == 'dofus'

    def test_bndbox(self):
        bndbox = next(iter(self.xml.tree.root.objects)).bndbox
        assert isinstance(bndbox, BNDBOX)
        assert bndbox.xmin == 97
        assert bndbox.ymin == 13
        assert bndbox.xmax == 203
        assert bndbox.ymax == 155

    def test_csv(self):
        csv, cols = self.xml.csv
        assert len(csv) == 1
        assert len(next(iter(csv))) == len(cols)

    def test_python(self):
        python = self.xml.python
        assert isinstance(python, dict)
        assert python.get('annotation').get('folder') == 'dofus'
        assert isinstance(python.get('annotation').get('object'), list)


class TestTFRecord(unittest.TestCase):
    xml = r'C:\Users\Minifranger\Documents\python_scripts\workout\test_workout\mocks\test_xml.xml'

    def setUp(self):
        self.tfrecord = TFRecord(xml=XML(path=self.xml))

    def test_tfrecord(self):
        assert isinstance(self.tfrecord, TFRecord)
        assert isinstance(self.tfrecord.tfrecord, dict)
        assert isinstance(self.tfrecord.tfrecord.get('filename'), tf.train.Feature)
        assert isinstance(self.tfrecord.tfrecord.get('name'), tf.train.Feature)