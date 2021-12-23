# from test_workout import TestTensorflow
# from workout.labelimg.data import Data
# from workout.vision.vision import Vision
#
#
# class TestDetection(TestTensorflow):
#     # fps = 8
#
#     def setUp(self):
#         super().setUp()
#         self.data = Data.factory(source=self.dofus)
#         self.vision = Vision.factory(source=self.dofus_graph)
#
#     def test_detect_image(self):
#         Vision.instance.detect_image(labels=Data.instance.labels_pbtxt,
#                                      image=r'C:\Users\Minifranger\Documents\python_scripts\workout\test_workout\test_vision\img.png')
#
#     def test_detect_video(self):
#         Vision.instance.detect_video(labels=Data.instance.labels_pbtxt, output='test_output.avi', title='Overwatch', fps=self.fps)
