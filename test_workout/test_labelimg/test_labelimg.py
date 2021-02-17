import os
import unittest

from workout.labelimg.labelimg import Train, Test, Data, Images, Labels, LabelIMG, TrainTFWriter, TestTFWriter, Image
from workout.labelimg.tfrecord import TFRecord, XML


class TestLabelIMG(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.path)

    def test_data(self):
        assert os.path.isdir(Data.instance.path)
        assert os.path.isdir(Images.instance.path)
        assert os.path.isdir(Labels.instance.path)
        assert os.path.isdir(Train.instance.path)
        assert os.path.isdir(Test.instance.path)

    def test_write(self):
        LabelIMG.write()
        assert os.path.isfile(TrainTFWriter.instance.path)
        assert os.path.isfile(TestTFWriter.instance.path)


        # import tensorflow as tf
        # example = tf.train.Example()
        # from tensorflow.python.lib.io.tf_record import tf_record_iterator
        # print('TRAIN RECORDS ')
        # for record in tf_record_iterator(str(TrainTFWriter.instance.path)):
        #     example.ParseFromString(record)
        #     f = example.features.feature
        #     print(f)
        # print('TEST RECORDS ')
        # for record in tf_record_iterator(str(TestTFWriter.instance.path)):
        #     example.ParseFromString(record)
        #     f = example.features.feature
        #     print(f)


class TestTrain(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.path)

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

    def test_train_write(self):
        Train.instance.write()
        assert os.path.isfile(TrainTFWriter.instance.path)
        assert os.path.getsize(TrainTFWriter.instance.path) > 0


class TestTest(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.path)

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

    def test_test_write(self):
        Test.instance.write()
        assert os.path.isfile(TestTFWriter.instance.path)
        assert os.path.getsize(TestTFWriter.instance.path) > 0


class TestImages(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.path)

    def test_images(self):
        images = Images.instance.images
        assert len(images) == 5
        assert all(isinstance(i, Image) for i in images)

    def test_format(self):
        images = Images.instance.images
        assert all(i.format.lower().endswith('.jpg') for i in images)

    def test_all(self):
        assert Images.instance.all
