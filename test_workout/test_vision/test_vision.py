from test_workout import TestTensorflow
from workout.labelimg.data import Data
from workout.vision.vision import Vision


class TestDetection(TestTensorflow):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    graph = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph'
    image = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1614284494123.jpg'

    def setUp(self):
        super().setUp()
        self.data = Data.factory(source=self.overwatch)
        self.vision = Vision.factory(source=self.graph)

    def test_detect_image(self):
        Vision.instance.detect_image(labels=Data.instance.labels_pbtxt, image=self.image)

