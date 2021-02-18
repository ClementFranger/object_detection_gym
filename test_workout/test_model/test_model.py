import os
import unittest

from workout.labelimg.labelimg import LabelIMG, Data
from workout.model.model import Model, TrainInput, PipelineConfig, Checkpoint, Training, Graph


class TestModel(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.overwatch)
        self.model = Model.factory(path=self.path)

    def test_train(self):
        Model.instance.train()

    def test_tensorboard(self):
        Model.instance.tensorboard()

    def test_save(self):
        Model.instance.save()


class TestPipelineConfig(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.model = Model.factory(path=self.path)

    def test_(self):
        config = Model.instance.pipeline_config
        assert isinstance(config, PipelineConfig)
        assert config.name == 'pipeline.config'
        assert os.path.isdir(config.source)
        assert os.path.isfile(config.path)

    #TODO : update config class to use training and checkpoint dir
    def test_update(self):
        Model.instance.config.update()
        assert os.path.isfile(Model.instance.config.config)


class TestCheckpoint(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.model = Model.factory(path=self.path)

    def test_(self):
        checkpoint = Model.instance.checkpoint
        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.name == 'checkpoint'
        assert os.path.isdir(checkpoint.source)

    def test_checkpoint(self):
        checkpoint = Model.instance.checkpoint.checkpoint


class TestTraining(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.model = Model.factory(path=self.path)

    def test_(self):
        training = Model.instance.training
        assert isinstance(training, Training)
        assert training.name == 'training'
        assert os.path.isdir(training.source)


class TestGraph(unittest.TestCase):
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        self.model = Model.factory(path=self.path)

    def test_(self):
        graph = Model.instance.graph
        assert isinstance(graph, Graph)
        assert graph.name == 'graph'
        assert os.path.isdir(graph.source)


class TestConfig(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    path = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'
    fine_tune_checkpoint = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\checkpoint\ckpt-0'
    num_classes = 1
    batch_size = 32
    num_steps = 10000

    def setUp(self):
        self.labelimg = LabelIMG.factory(path=self.overwatch)
        self.model = Model.factory(path=self.path, num_classes=self.num_classes, batch_size=self.batch_size,
                                   num_steps=self.num_steps, data=Data.instance,
                                   fine_tune_checkpoint=self.fine_tune_checkpoint)





    def test_(self):
        import cv2
        import tensorflow as tf
        from tensorflow.core.framework.graph_pb2 import GraphDef
        from tensorflow.python.saved_model import loader
        from tensorflow.python.client.session import Session
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils

        # label_map = label_map_util.load_labelmap(TrainInput.instance.labels)
        # categories = label_map_util.convert_label_map_to_categories(
        #     label_map, max_num_classes=1, use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)

        category_index = label_map_util.create_category_index_from_labelmap(TrainInput.instance.labels, use_display_name=True)
        graph = tf.saved_model.load(r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph\saved_model', tags=None, options=None)
        infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        image_np = np.array(Image.open((r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\71.jpg')))

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = infer(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        img = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        im = Image.fromarray(img)
        im.save("your_file.jpeg")
        # plt.figure()
        # plt.imshow(image_np_with_detections)
        # plt.show()
