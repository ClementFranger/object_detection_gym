import time
import logging
from functools import wraps
import numpy as np
from tensorflow import Tensor
from object_detection.utils import visualization_utils

from workout.image import Image

logger = logging.getLogger(__name__)


class Detection:

    class Decorators:
        @classmethod
        def time(cls, f):
            @wraps(f)
            def wrapper(self, *args, **kwargs):
                start_time = time.time()
                result = f(self, *args, **kwargs)
                end_time = time.time()
                logger.info('{name} function took {time} seconds'.format(name=f.__name__, time=round(end_time - start_time, 2)))
                return result
            return wrapper

    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.image = kwargs.get('image')
        assert isinstance(self.image, Image)
        self.detections = self._clean(detections=self.detect())

    @property
    def num_detections(self):
        return self.detections.get('num_detections')

    @property
    def detection_classes(self):
        return self.detections.get('detection_classes')

    @property
    def detection_scores(self):
        return self.detections.get('detection_scores')

    @property
    def detection_boxes(self):
        return self.detections.get('detection_boxes')

    @property
    def relevant_detections(self):
        return

    def _clean(self, **kwargs):
        detections = kwargs.get('detections')
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return detections

    @Decorators.time
    def detect(self):
        return self.model(self.image.tensor)

    @Decorators.time
    def draw_boxes(self, **kwargs):
        category_index = kwargs.get('category_index')
        min_score_thresh = kwargs.get('min_score_thresh', 0.30)
        return visualization_utils.visualize_boxes_and_labels_on_image_array(
            self.image.cleaned, self.detection_boxes, self.detection_classes, self.detection_scores,
            category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=min_score_thresh,
            agnostic_mode=False)
