import logging
import numpy as np
from object_detection.utils import visualization_utils

from workout.image import Image

logger = logging.getLogger(__name__)


class Detection:
    instance = None

    def __init__(self, **kwargs):
        self.model = None
        self.image = None
        """ result of tensorflow model detection on image """
        self.detections = None

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

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

    def _update_inputs(self, **kwargs):
        self.model = kwargs.get('model')
        self.image = kwargs.get('image')
        assert isinstance(self.image, Image)

    def detect(self, **kwargs):
        if kwargs.get('model') and kwargs.get('image'):
            self._update_inputs(**kwargs)
        detections = self.model(self.image.tensor)
        self.detections = detections if kwargs.get('clean', False) else self._clean(detections=detections)
        return self.detections

    def draw_boxes(self, **kwargs):
        category_index = kwargs.get('category_index')
        min_score_thresh = kwargs.get('min_score_thresh', 0.30)
        return visualization_utils.visualize_boxes_and_labels_on_image_array(
            self.image.cleaned, self.detection_boxes, self.detection_classes, self.detection_scores,
            category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=min_score_thresh,
            agnostic_mode=False)
