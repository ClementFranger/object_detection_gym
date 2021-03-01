import tensorflow as tf
from workout.utils import Image, MSSImage


class ImageTensor:
    def __init__(self, **kwargs):
        self.image = kwargs.get('image')
        print(type(self.image))
        # assert isinstance(self.image, Image) or isinstance(self.image, MSSImage)
        # self.image = self.image.cleaned

    @property
    def tensor(self):
        input_tensor = tf.convert_to_tensor(self.image)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor
