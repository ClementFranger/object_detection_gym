import logging
import tensorflow as tf
from workout.utils import Source

logger = logging.getLogger(__name__)


class Model(Source):
    name = 'saved_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.saved_model.load(str(self.path), tags=None, options=None)
