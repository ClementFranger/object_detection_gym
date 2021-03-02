import os
import logging
import cv2
import numpy as np
import PIL
import tensorflow as tf

from mss.screenshot import ScreenShot
from object_detection.utils import visualization_utils
from tensorflow.python.platform.gfile import GFile

from workout.record.application import Application

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

    # @property
    # def tensor(self):
    #     tensor = tf.convert_to_tensor(self.np_array)
    #     tensor = tensor[tf.newaxis, ...]
    #     return tensor

    # def draw_boxes(self, **kwargs):
    #     detections, category_index = kwargs.get('detections'), kwargs.get('category_index')
    #     min_score_thresh = kwargs.get('min_score_thresh', 0.30)
    #     assert isinstance(detections, Detection)
    #     return visualization_utils.visualize_boxes_and_labels_on_image_array(
    #         self.np_array, detections.detection_boxes, detections.detection_classes, detections.detection_scores,
    #         category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=min_score_thresh,
    #         agnostic_mode=False)


class MSSImage(Image):
    def __init__(self, **kwargs):
        self.image = kwargs.get('image')
        assert isinstance(self.image, ScreenShot)
        # self.cleaned = cv2.resize(self.np_array, (1920, 1080))
        self.cleaned = self.clean()

    def clean(self):
        # logger.info('cleaning frame')
        image = self._resize(image=self.np_array)
        # image = self._color(image=image)
        return image

    @property
    def np_array(self):
        image = np.array(self.image)
        image = np.flip(image[:, :, :3], 2)
        return image

    def _resize(self, **kwargs):
        # logger.info('Resizing image to {size}'.format(size=Application.instance.size))
        return cv2.resize(kwargs.get('image'), Application.instance.size)

    # def _color(self, **kwargs):
    #     # image = cv2.cvtColor(kwargs.get('image'), cv2.COLOR_RGB2BGR)
    #     # image = cv2.cvtColor(kwargs.get('image'), cv2.COLOR_BGR2RGB)
    #     # return image
    #     return kwargs.get('image')

    # def draw_boxes(self, **kwargs):
    #     detections, category_index = kwargs.get('detections'), kwargs.get('category_index')
    #     min_score_thresh = kwargs.get('min_score_thresh', 0.30)
    #     assert isinstance(detections, Detection)
    #     return visualization_utils.visualize_boxes_and_labels_on_image_array(
    #         self.cleaned, detections.detection_boxes, detections.detection_classes, detections.detection_scores,
    #         category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=min_score_thresh,
    #         agnostic_mode=False)
