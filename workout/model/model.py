# # Some models to train on
# MODELS_CONFIG = {
#     'ssd_mobilenet_v3': {
#         'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
#     },
#     'faster_rcnn_inception_v2': {
#         'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
#     },
# }
#
# # Select a model from `MODELS_CONFIG`.
# # I chose ssd_mobilenet_v2 for this project, you could choose any
# selected_model = 'ssd_mobilenet_v3'


class Model:

    def __init__(self, **kwargs):
        Config.factory(path=kwargs.get('config'), **kwargs)

    @property
    def config(self):
        return Config.instance


class Config:
    instance = None

    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        self.num_class = 1
        self.checkpoint = None
        self.num_steps = None

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

class SSD:
    pass
class TrainConfig:
    pass
class TestConfig:
    pass
class TrainInput:
    pass
class TestInput:
    pass