import os
import logging
import cv2
import numpy as np
import PIL
from pathlib import Path

from mss.screenshot import ScreenShot
from object_detection.utils import visualization_utils
from tensorflow.python.platform.gfile import GFile

from workout.record.application import Application
from workout.vision.detection import Detection

logger = logging.getLogger(__name__)


class Schema:

    @classmethod
    def keys(cls):
        return [k for k in vars(cls) if not k.startswith('__')]

    @classmethod
    def values(cls):
        return [getattr(cls, k) for k in cls.keys()]


class Source:
    instance = None
    name = None

    def __init__(self, **kwargs):
        self.source, self.name = kwargs.get('source'), kwargs.get('name') or self.name
        self.path = Path(kwargs.get('path') or os.path.join(self.source, self.name))
        if not os.path.exists(self.path):
            logger.warning('{path} does not exist. Creating one'.format(path=self.path))
            self.path.mkdir(parents=True, exist_ok=True)
        logger.info('Creating {cls} factory from {path}'.format(cls=self.__class__, path=self.path))

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance


# TODO : refacto image classes
class Image:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')

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

    def draw_boxes(self, **kwargs):
        detections, category_index = kwargs.get('detections'), kwargs.get('category_index')
        min_score_thresh = kwargs.get('min_score_thresh', 0.30)
        assert isinstance(detections, Detection)
        return visualization_utils.visualize_boxes_and_labels_on_image_array(
            self.np_array, detections.detection_boxes, detections.detection_classes, detections.detection_scores,
            category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=min_score_thresh,
            agnostic_mode=False)


class MSSImage:
    def __init__(self, **kwargs):
        self.image = kwargs.get('image')
        assert isinstance(self.image, ScreenShot)

    @property
    def cleaned(self):
        image = self._resize(image=self.np_array)
        image = self._color(image=image)
        return image

    @property
    def np_array(self):
        return np.array(self.image)

    def _resize(self, **kwargs):
        return cv2.resize(kwargs.get('image'), Application.instance.size)

    def _color(self, **kwargs):
        image = cv2.cvtColor(kwargs.get('image'), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def draw_boxes(self, **kwargs):
        detections, category_index = kwargs.get('detections'), kwargs.get('category_index')
        min_score_thresh = kwargs.get('min_score_thresh', 0.30)
        assert isinstance(detections, Detection)
        return visualization_utils.visualize_boxes_and_labels_on_image_array(
            self.cleaned, detections.detection_boxes, detections.detection_classes, detections.detection_scores,
            category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=min_score_thresh,
            agnostic_mode=False)