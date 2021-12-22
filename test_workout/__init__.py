import unittest
import tensorflow as tf


class TestTensorflow(unittest.TestCase):
    overwatch = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch'
    overwatch_model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'
    dofus = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus'
    dofus_model = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\models\ssd_mobilenet_v2_320x320_coco17_tpu-8'

    def setUp(self):
        """ IMPORTANT : so that CUDNN does not fail loading """
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
