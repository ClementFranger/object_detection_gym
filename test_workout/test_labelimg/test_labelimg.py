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







