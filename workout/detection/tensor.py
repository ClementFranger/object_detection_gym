import tensorflow as tf
from workout.utils import Image


class ImageTensor:
    def __init__(self, **kwargs):
        self.image = kwargs.get('image')
        assert isinstance(self.image, Image)

    @property
    def tensor(self):
        print('convert to tensor')
        input_tensor = tf.convert_to_tensor(self.image.np_array)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor
