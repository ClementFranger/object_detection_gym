import logging
import tensorflow as tf
from workout.utils import Source

logger = logging.getLogger(__name__)


class TrainedModel(Source):
    name = 'saved_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.saved_model.load(str(self.path), tags=None, options=None)
        """ important to set signature for the run function """
        self.model_with_signatures = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

