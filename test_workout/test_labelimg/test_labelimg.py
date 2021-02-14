import os
import unittest

from workout.labelimg.labelimg import Train, Test, Data, Images, Labels, LabelIMG, TrainTFWriter, TestTFWriter
from workout.labelimg.tfrecord import TFRecord, XML


class TestLabelIMG(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.path)

    def test_data(self):
        assert os.path.isdir(Data.instance.path)
        assert os.path.isdir(Images.instance.path)
        assert os.path.isdir(Labels.instance.path)
        assert os.path.isdir(Train.instance.path)
        assert os.path.isdir(Test.instance.path)


class TestTrain(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus'

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
        # """ do not run multiple times as it will append to current record """
        # TrainTFWriter.factory()
        # Train.instance.write_tfrecord(writer=TrainTFWriter.instance.writer)
        Train.instance.write()
        assert os.path.isfile(TrainTFWriter.instance.path)

        # import tensorflow as tf
        # example = tf.train.Example()
        # from tensorflow.python.lib.io.tf_record import tf_record_iterator
        # for record in tf_record_iterator(r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\data\train.record'):
        #     example.ParseFromString(record)
        #     f = example.features.feature
        #     print(f)
            # v1 = f['int64 feature'].int64_list.value[0]
            # v2 = f['float feature'].float_list.value[0]
            # v3 = f['bytes feature'].bytes_list.value[0]
            # print(v1, v2, v3)


class TestTest(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus'

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



