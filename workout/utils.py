import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Schema:

    @classmethod
    def keys(cls):
        return [k for k in vars(cls)]

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
