# import tensorflow as tf
# from numpy.core.multiarray import ndarray
#
#
# class ImageTensor:
#     def __init__(self, **kwargs):
#         self.image = kwargs.get('image')
#         assert isinstance(self.image, ndarray)
#
#     @property
#     def tensor(self):
#         input_tensor = tf.convert_to_tensor(self.image)
#         input_tensor = input_tensor[tf.newaxis, ...]
#         return input_tensor
