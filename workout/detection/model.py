import os
import tensorflow as tf
from pathlib import Path


class Model:
    instance = None

    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        assert os.path.isdir(self.path)
        # self.model = self.load()

        # graph = tf.saved_model.load(r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph\saved_model', tags=None, options=None)
        # infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        # self.model = infer

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    # def load(self):
    #     print('loading from %s' % str(self.path))
    #     graph = tf.saved_model.load(str(self.path), tags=None, options=None)
    #     infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    #     return infer

    # def detect(self, **kwargs):
    #     return self.model(**kwargs)