import os
import logging
import cv2
import numpy as np
import PIL
import tensorflow as tf

from mss.screenshot import ScreenShot
from tensorflow.python.platform.gfile import GFile

logger = logging.getLogger(__name__)


class Image:

    @property
    def tensor(self):
        tensor = tf.convert_to_tensor(self.cleaned)
        tensor = tensor[tf.newaxis, ...]
        return tensor


class PathImage(Image):
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        assert os.path.isfile(self.path)
        self.cleaned = self.np_array

    @property
    def encoded(self):
        with GFile(self.path, 'rb') as f:
            encoded = f.read()
        return encoded

    @property
    def format(self):
        _, format = os.path.splitext(self.path)
        return format

    @property
    def np_array(self):
        return np.array(PIL.Image.open((self.path)))


class MSSImage(Image):
    def __init__(self, **kwargs):
        self.image = kwargs.get('image')
        self.application = kwargs.get('application')
        assert isinstance(self.image, ScreenShot)
        self.cleaned = self.clean()

    def clean(self):
        image = self._resize(image=self.np_array)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @property
    def np_array(self):
        image = np.array(self.image)
        image = np.flip(image[:, :, :3], 2)
        return image

    def _resize(self, **kwargs):
        return cv2.resize(kwargs.get('image'), self.application.size)

