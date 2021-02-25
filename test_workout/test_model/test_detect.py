import os
import unittest

from workout.labelimg.labelimg import LabelIMG, Data
from workout.model.detection import DetectionModelInterface, ImageTensor, DetectionModel
from workout.model.model import Model, TrainInput, PipelineConfig, Checkpoint, Training, Graph


class TestDetectionModel(unittest.TestCase):
    # overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'
    image = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1.jpg'

    def setUp(self):
        # self.labelimg = LabelIMG.factory(path=self.overwatch)
        self.model = Model.factory(model=self.model)
        self.detection_model_interface = DetectionModelInterface.factory(source=Model.instance.graph.path)

    def test_detect(self):
        detect = DetectionModelInterface.instance.detection_model.detect(input_tensor=ImageTensor(image=self.image))
        print(detect)