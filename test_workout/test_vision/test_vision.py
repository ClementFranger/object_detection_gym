from test_workout import TestTensorflow
from workout.labelimg.data import Data
from workout.vision.vision import Vision


class TestDetection(TestTensorflow):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    graph = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph'
    image = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1614284550307.jpg'
    fps = 5

    def setUp(self):
        super().setUp()
        self.data = Data.factory(source=self.overwatch)
        self.vision = Vision.factory(source=self.graph)

    def test_detect_image(self):
        Vision.instance.detect_image(labels=Data.instance.labels_pbtxt, image=self.image)

    def test_detect_video(self):
        Vision.instance.detect_video(labels=Data.instance.labels_pbtxt, output='test_output.avi', title='Overwatch', fps=self.fps)
