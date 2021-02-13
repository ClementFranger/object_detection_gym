import os
import unittest

from workout.labelimg import Train, Test, Data, Images, Labels, LabelIMG


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
