import os
from workout.utils import Source


class Graph(Source):
    """ directory containing saved model """
    name = 'graph'

    @property
    def saved_model(self):
        return os.path.join(self.path, 'saved_model')
