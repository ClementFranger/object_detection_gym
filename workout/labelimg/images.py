import logging

from workout.image import PathImage
from workout.utils import Source

logger = logging.getLogger(__name__)


class Images(Source):
    """ directory containing all images data """
    name = 'images'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.all

    @property
    def images(self):
        images = []
        for i in list(self.path.glob('*.jpg')):
            images.append(PathImage(path=i))
        return images

    @property
    def all(self, **kwargs):
        return all([i.format.lower().endswith(kwargs.get('format', 'jpg')) for i in self.images])
