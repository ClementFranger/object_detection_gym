import unittest
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)


class TestTensorflow(unittest.TestCase):
    data = r'C:\Users\Minifranger\Documents\python_scripts\percepteur\percepteur\data'
    model = r'C:\Users\Minifranger\Documents\python_scripts\percepteur\percepteur\data\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'
    graph = r'C:\Users\Minifranger\Documents\python_scripts\percepteur\percepteur\data\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph'

    def setUp(self):
        """ IMPORTANT : so that CUDNN does not fail loading """
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)