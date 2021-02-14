import os
import unittest
import tensorflow as tf

from workout.labelimg.tfrecord import XML, Tree, Root, Size, BNDBOX, TFRecord


class TestXML(unittest.TestCase):
    xml = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data\labels\1.xml'

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
    xml = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data\labels\1.xml'

    def setUp(self):
        self.tfrecord = TFRecord(xml=XML(path=self.xml))

    def test_tfrecord(self):
        assert isinstance(self.tfrecord, TFRecord)
        assert isinstance(self.tfrecord.tfrecord, dict)
        assert isinstance(self.tfrecord.tfrecord.get('filename'), tf.train.Feature)
        assert isinstance(self.tfrecord.tfrecord.get('name'), tf.train.Feature)