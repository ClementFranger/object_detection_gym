import time
import logging
from functools import wraps
import numpy as np
from tensorflow import Tensor


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
                logger.info('{name} function took {time} seconds'.format(name=f.__name__, time=end_time - start_time))
                return result
            return wrapper

    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.tensor = kwargs.get('tensor')
        assert isinstance(self.tensor, Tensor)
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
        return self.model(self.tensor)
