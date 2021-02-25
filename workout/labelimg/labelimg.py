# import os
# import logging
# from pathlib import Path
# from tensorflow.python.platform.gfile import GFile
#
# from workout.labelimg.tfrecord import XML
# from workout.labelimg.tfwriter import TrainTFWriter, TestTFWriter
# from workout.utils import Image, Source
#
# logger = logging.getLogger(__name__)
#
#
# class LabelIMG:
#     instance = None
#
#     def __init__(self, **kwargs):
#         logger.info('Creating labelimg factory from {path}'.format(path=kwargs.get('path')))
#         Data.factory(path=Path(os.path.join(kwargs.get('path'), kwargs.get('data', Data.name))))
#         Images.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('images', Images.name))))
#         Labels.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('labels', Labels.name))))
#         Train.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('train', Train.name))))
#         Test.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('test', Test.name))))
#
#     @classmethod
#     def factory(cls, **kwargs):
#         if cls.instance is None:
#             cls.instance = cls(**kwargs)
#         assert isinstance(cls.instance, cls)
#         return cls.instance
#
#     @property
#     def data(self):
#         return Data.instance
#
#     @staticmethod
#     def write():
#         Train.instance.write()
#         Test.instance.write()
#
#
# # class Source:
# #     instance = None
# #     name = None
# #
# #     def __init__(self, **kwargs):
# #         self.path = Path(kwargs.get('path'))
# #
# #     @classmethod
# #     def factory(cls, **kwargs):
# #         if cls.instance is None:
# #             cls.instance = cls(**kwargs)
# #         assert isinstance(cls.instance, cls)
# #         return cls.instance
#
#
#
#
#
#
#
#
#
#
#
#
#
#
