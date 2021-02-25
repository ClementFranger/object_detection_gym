import os
from workout.utils import Source


class Checkpoint(Source):
    """ directory containing pretrained model checkpoint """
    name = 'checkpoint'

    @property
    def checkpoint(self):
        return os.path.join(self.path, 'ckpt-0')
