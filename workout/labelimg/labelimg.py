import os
import logging
from pathlib import Path
from tensorflow.python.platform.gfile import GFile

from workout.labelimg.tfrecord import XML
from workout.labelimg.tfwriter import TrainTFWriter, TestTFWriter

logger = logging.getLogger(__name__)


class LabelIMG:
    instance = None

    def __init__(self, **kwargs):
        logger.info('Creating labelimg factory from {path}'.format(path=kwargs.get('path')))
        Data.factory(path=Path(os.path.join(kwargs.get('path'), kwargs.get('data', Data.name))))
        Images.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('images', Images.name))))
        Labels.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('labels', Labels.name))))
        Train.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('train', Train.name))))
        Test.factory(path=Path(os.path.join(Data.instance.path, kwargs.get('test', Test.name))))

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def data(self):
        return Data.instance

    @staticmethod
    def write():
        Train.instance.write()
        Test.instance.write()


class Source:
    instance = None
    name = None

    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def xml(self):
        xml = []
        logger.info('Retrieving {name} data as xml'.format(name=self.path.name))
        for x in list(self.path.glob('*.xml')):
            xml.append(XML(path=x, image=Image()))
        return xml

    @property
    def tfrecord(self):
        tfrecord = []
        logger.info('Retrieving {name} data as tfrecord'.format(name=self.path.name))
        for x in list(self.path.glob('*.xml')):
            tfrecord.append(XML(path=x, image=Image(
                path=os.path.join(Images.instance.path, x.name.replace('xml', 'jpg')))).tfrecord)
        return tfrecord


class Data(Source):
    name = 'data'

    @property
    def images(self):
        return Images.instance

    @property
    def labels(self):
        return Labels.instance

    @property
    def train(self):
        return Train.instance

    @property
    def test(self):
        return Test.instance


class Images(Source):
    name = 'images'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.all

    @property
    def images(self):
        images = []
        logger.info('Retrieving {name} data as image'.format(name=self.path.name))
        for i in list(self.path.glob('*.jpg')):
            images.append(Image(path=i))
        return images

    @property
    def all(self, **kwargs):
        return all([i.format.lower().endswith(kwargs.get('format', 'jpg')) for i in self.images])

    def split(self):
        return


class Labels(Source):
    name = 'labels'
    pass


class Train(Source):
    name = 'train'

    @property
    def writer(self):
        TrainTFWriter.factory(path=os.path.join(Data.instance.path, '{name}.record'.format(name=self.name)))
        return TrainTFWriter.instance

    def write(self):
        for r in self.tfrecord:
            self.writer.write(tfrecord=r.tfrecord)


class Test(Source):
    name = 'test'

    @property
    def writer(self):
        TestTFWriter.factory(path=os.path.join(Data.instance.path, '{name}.record'.format(name=self.name)))
        return TestTFWriter.instance

    def write(self):
        for r in self.tfrecord:
            self.writer.write(tfrecord=r.tfrecord)


class Image:
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')

    @property
    def encoded(self):
        with GFile(self.path, 'rb') as f:
            encoded = f.read()
        return encoded

    @property
    def format(self):
        _, format = os.path.splitext(self.path)
        return format
