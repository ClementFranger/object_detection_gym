import os
import unittest
from collections import namedtuple

from workout.labelimg import Train, Test, Data, Images, Labels, LabelIMG, XML, Root, Tree, Size, BNDBOX


class TestLabelIMG(unittest.TestCase):
    data = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data'

    def setUp(self):
        self.dofus = LabelIMG.factory(path=self.data)

    def test_data(self):
        assert os.path.isdir(Data.instance.path)
        assert os.path.isdir(Images.instance.path)
        assert os.path.isdir(Labels.instance.path)
        assert os.path.isdir(Train.instance.path)
        assert os.path.isdir(Test.instance.path)

    def test_data_csv(self):
        Data.instance.csv()
        assert os.path.isfile(os.path.join(Data.instance.path, '{name}.csv'.format(name=Train.instance.path.name)))
        assert os.path.isfile(os.path.join(Data.instance.path, '{name}.csv'.format(name=Test.instance.path.name)))

    def test_data_pbtxt(self):
        Data.instance.pbtxt()
        assert os.path.isfile(os.path.join(Data.instance.path, '{name}.pbtxt'.format(name=Data.instance.path.name)))

    def test_(self):
        import tensorflow as tf
        import pandas as pd

        writer = tf.io.TFRecordWriter(os.path.join(Data.instance.path, 'train.record'))
        # path = os.path.join(image_dir)
        examples = pd.read_csv(os.path.join(Data.instance.path, 'train.csv'))

        def split(df, group):
            data = namedtuple('data', ['filename', 'object'])
            gb = df.groupby(group)
            print('groups keys are %s' % gb.groups.keys())
            print('groups are %s' % gb.groups)
            return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

        print(examples)
        grouped = split(examples, 'filename')
        for g in grouped:
            print('------------------------')
            print('printing data')
            print(g.filename)
            print('printing groupby')
            print(g.object)
            print('------------------------')
            # tf_example = create_tf_example(group, path)
            # writer.write(tf_example.SerializeToString())


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
