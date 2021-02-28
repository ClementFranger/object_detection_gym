import logging

import PIL
import tensorflow as tf
from object_detection.utils import label_map_util

from workout.vision.detection import Detection
from workout.vision.model import Model
from workout.vision.tensor import ImageTensor
from workout.utils import Image

logger = logging.getLogger(__name__)


# TODO : solve this session problem so that we can load model from another class. For now do everything here
class Vision:
    instance = None

    def __init__(self, **kwargs):
        Model.factory(**kwargs)

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def model(self):
        return Model.instance

    def model_with_signatures(self):
        """ important to set signature in the run function """
        return self.model.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def category_index(self, **kwargs):
        return label_map_util.create_category_index_from_labelmap(kwargs.get('labels'), use_display_name=True)

    def detect_image(self, **kwargs):
        category_index = self.category_index(**kwargs)
        model, image = self.model_with_signatures(), Image(path=kwargs.get('image'))
        detections = Detection(model=model, tensor=ImageTensor(image=image).tensor, **kwargs)
        img = image.draw_boxes(detections=detections, category_index=category_index)

        im = PIL.Image.fromarray(img)
        im.save("your_file.jpeg")


