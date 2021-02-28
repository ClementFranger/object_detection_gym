import os
import unittest
import tensorflow as tf

from test_workout import TestTensorflow
from workout.vision.model import Model
from workout.labelimg.data import Data
from workout.vision.vision import Vision


class TestDetection(TestTensorflow):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    graph = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph'

    def setUp(self):
        super().setUp()
        self.data = Data.factory(source=self.overwatch)
        self.vision = Vision.factory(source=self.graph)

    # def test_(self):
    #     assert isinstance(Vision.instance, Vision)
    #     assert Vision.instance.name == 'saved_model'
    #     assert os.path.isdir(Vision.instance.source)
    #     assert os.path.isdir(Vision.instance.path)

    # def test_model(self):
    #     assert isinstance(Vision.instance.model, Model)
    #     assert os.path.isdir(Vision.instance.model.path)

    def test_detect_image(self):
        Vision.instance.detect_image(labels=Data.instance.labels_pbtxt, image=r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1614284494123.jpg')

    # def test_detect_image2(self):
    #     Detection.instance.detect_image2(labels=Data.instance.labels_pbtxt)
    #
    # def return_graph(self):
    #     import tensorflow as tf
    #
    #     sess = tf.compat.v1.Session()
    #     graph = tf.saved_model.load(r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph\saved_model', tags=None, options=None)
    #     infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #     return infer, sess

    # def test_infere(self):
    #     # self.data = Data.factory(source=self.overwatch)
    #     # self.detection = Detection.factory(source=self.graph)
    #     import cv2
    #     import tensorflow as tf
    #     from tensorflow.core.framework.graph_pb2 import GraphDef
    #     from tensorflow.python.saved_model import loader
    #     from tensorflow.python.client.session import Session
    #     from object_detection.utils import label_map_util
    #     from object_detection.utils import visualization_utils as viz_utils
    #
    #     # label_map = label_map_util.load_labelmap(TrainInput.instance.labels)
    #     # categories = label_map_util.convert_label_map_to_categories(
    #     #     label_map, max_num_classes=1, use_display_name=True)
    #     # category_index = label_map_util.create_category_ind
    #     #         print(type(infer))ex(categories)
    #
    #     category_index = label_map_util.create_category_index_from_labelmap(Data.instance.labels_pbtxt, use_display_name=True)
    #     # graph = tf.saved_model.load(r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph\saved_model', tags=None, options=None)
    #     # infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #     global sess
    #     global infer
    #     infer, sess = self.return_graph()
    #     # tf.compat.v1.keras.backend.set_session(sess)
    #
    #     import numpy as np
    #     from PIL import Image
    #     import matplotlib.pyplot as plt
    #     image_np = np.array(Image.open((r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1614284494123.jpg')))
    #
    #     # Things to try:
    #     # Flip horizontally
    #     # image_np = np.fliplr(image_np).copy()
    #
    #     # Convert image to grayscale
    #     # image_np = np.tile(
    #     #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    #
    #     # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    #     input_tensor = tf.convert_to_tensor(image_np)
    #     # The model expects a batch of images, so add an axis with `tf.newaxis`.
    #     input_tensor = input_tensor[tf.newaxis, ...]
    #
    #     # input_tensor = np.expand_dims(image_np, 0)
    #     detections = infer(input_tensor)
    #
    #     # All outputs are batches tensors.
    #     # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    #     # We're only interested in the first num_detections.
    #     num_detections = int(detections.pop('num_detections'))
    #     # print('num detections %s' % num_detections)
    #     # print('detections length %s' % len(detections))
    #     # print('detections keys %s' % detections.keys())
    #     # print('detections anchor indices %s' % detections.get('detection_boxes'))
    #     detections = {key: value[0, :num_detections].numpy()
    #                   for key, value in detections.items()}
    #     # detections = {key: value[0] if isinstance(value, list) else np.array([value[0]])
    #     #               for key, value in detections.items()}
    #     # print(detections)
    #     detections['num_detections'] = num_detections
    #
    #     # detection_classes should be ints.
    #     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    #
    #     # print('detections anchor indices %s' % detections.get('detection_boxes'))
    #     image_np_with_detections = image_np.copy()
    #
    #     img = viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes'],
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=200,
    #         min_score_thresh=.30,
    #         agnostic_mode=False)
    #
    #     im = Image.fromarray(img)
    #     im.save("your_file.jpeg")
    #     # plt.figure()
    #     # plt.imshow(image_np_with_detections)
    #     # plt.show()
    #
    # def test_infere2(self):
    #     import numpy as np
    #     from PIL import Image
    #     import tensorflow as tf
    #     from object_detection.utils import label_map_util
    #     from object_detection.utils import visualization_utils as viz_utils
    #
    #     category_index = label_map_util.create_category_index_from_labelmap(Data.instance.labels_pbtxt, use_display_name=True)
    #     graph = tf.saved_model.load(r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph\saved_model', tags=None, options=None)
    #     infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #
    #     image_np = np.array(Image.open((r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1614284494123.jpg')))
    #
    #     input_tensor = tf.convert_to_tensor(image_np)
    #     input_tensor = input_tensor[tf.newaxis, ...]
    #
    #     detections = infer(input_tensor)
    #
    #     num_detections = int(detections.pop('num_detections'))
    #     detections = {key: value[0, :num_detections].numpy()
    #                   for key, value in detections.items()}
    #     detections['num_detections'] = num_detections
    #
    #     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    #
    #     image_np_with_detections = image_np.copy()
    #
    #     img = viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes'],
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=200,
    #         min_score_thresh=.30,
    #         agnostic_mode=False)
    #
    #     im = Image.fromarray(img)
    #     im.save("your_file.jpeg")

