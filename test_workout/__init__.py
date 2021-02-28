import unittest
import tensorflow as tf


class TestTensorflow(unittest.TestCase):

    def setUp(self):
        """ IMPORTANT : so that CUDNN does not fail loading """
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
