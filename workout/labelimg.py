import os
import logging
import pandas
from functools import wraps
from pathlib import Path
from xml.etree import ElementTree

from workout.utils import Schema

logger = logging.getLogger(__name__)


class LabelIMG:
    instance = None
    data = None

    class Decorators:
        @classmethod
        def csv(cls, **kwargs):
            def decorator(f):
                @wraps(f)
                def wrapper(self, *args, **kwargs):
                    csv, cols = f(self, *args, **kwargs)
                    d = kwargs.get('d')
                    path = os.path.join(d, '{name}.csv'.format(name=d.name))
                    logger.info('Outputing data to {csv}'.format(csv=path))
                    return pandas.DataFrame(csv, columns=cols).to_csv(path)
                return wrapper
            return decorator

    @classmethod
    def factory(cls):
        if cls.instance is None:
            cls.instance = cls()
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def labels(self):
        return Path(os.path.join(self.data, 'labels'))

    @property
    def images(self):
        return Path(os.path.join(self.data, 'images'))

    @property
    def train(self):
        return Path(os.path.join(self.data, 'train'))

    @property
    def test(self):
        return Path(os.path.join(self.data, 'test'))

    @Decorators.csv()
    def csv(self, **kwargs):
        """ takes a directory and convert all xml files to csv """
        d, csv = kwargs.get('d'), []
        for xml in list(Path(d).glob('*.xml')):
            csv.extend(XML(path=xml).csv)
        return csv, XML.LabelIMGSchema.values()

    def data_to_csv(self):
        for d in [self.train, self.test]:
            logger.info('Converting {d} data to csv'.format(d=d))
            self.csv(d=d)


class XML:
    class XMLSchema(Schema):
        OBJECT = 'object'
        SIZE = 'size'
        BNDBOX = 'bndbox'

    class LabelIMGSchema(Schema):
        FILENAME = 'filename'
        WIDTH = 'width'
        HEIGHT = 'height'
        NAME = 'name'
        XMIN = 'xmin'
        YMIN = 'ymin'
        XMAX = 'xmax'
        YMAX = 'ymax'

    def __init__(self, **kwargs):
        self.path = kwargs.get('path')

    def _csv(self):
        csv = []
        root = ElementTree.parse(self.path).getroot()
        for o in root.findall(self.XMLSchema.OBJECT):
            csv.append((root.find(self.LabelIMGSchema.FILENAME).text,
                        int(root.find(self.XMLSchema.SIZE).find(self.LabelIMGSchema.WIDTH).text),
                        int(root.find(self.XMLSchema.SIZE).find(self.LabelIMGSchema.HEIGHT).text),
                        o.find(self.LabelIMGSchema.NAME).text,
                        int(o.find(self.XMLSchema.BNDBOX).find(self.LabelIMGSchema.XMIN).text),
                        int(o.find(self.XMLSchema.BNDBOX).find(self.LabelIMGSchema.YMIN).text),
                        int(o.find(self.XMLSchema.BNDBOX).find(self.LabelIMGSchema.XMAX).text),
                        int(o.find(self.XMLSchema.BNDBOX).find(self.LabelIMGSchema.YMAX).text)))
        return csv

    @property
    def csv(self):
        logger.info('Converting {path} to csv'.format(path=self.path))
        return self._csv()
