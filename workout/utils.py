import os
import logging
import numpy as np
import PIL
from pathlib import Path
from object_detection.utils import visualization_utils
from tensorflow.python.platform.gfile import GFile

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
