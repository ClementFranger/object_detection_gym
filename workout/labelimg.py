import os
import logging
import pandas
import tensorflow as tf
from functools import wraps
from pathlib import Path
from xml.etree import ElementTree
from object_detection.utils import dataset_util

from workout.utils import Schema

logger = logging.getLogger(__name__)


class LabelIMG:
    instance = None

    def __init__(self, **kwargs):
        logger.info('Creating data factory from {path}'.format(path=kwargs.get('path')))
        Data.factory(**kwargs)

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def data(self):
        return Data.instance


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

    # @property
    # def files(self):
    #     files = list(self.path.glob('*.xml'))
    #     logger.info('Found {count} xml files in {name}'.format(count=len(files), name=self.path.name))
    #     return files

    @property
    def xml(self):
        xml = []
        logger.info('Retrieving {name} data as xml'.format(name=self.path.name))
        for x in list(self.path.glob('*.xml')):
            xml.append(XML(path=x))
        return xml

    @property
    def tfrecord(self):
        tfrecord = []
        logger.info('Retrieving {name} data as tfrecord'.format(name=self.path.name))
        for x in list(self.path.glob('*.xml')):
            tfrecord.append(XML(path=x).tfrecord)
        return tfrecord

    def write_tfrecord(self):
        for r in self.tfrecord:
            r.write()
    # @property
    # def csv(self):
    #     csv = []
    #     logger.info('Retrieving {name} data as csv'.format(name=self.path.name))
    #     for xml in self.files:
    #         csv.extend(XML(path=xml).csv)
    #     return csv, XML.LabelIMGSchema.values()

    # @property
    # def pbtxt(self):
    #     pbtxt = []
    #     logger.info('Retrieving {name} data as pbtxt'.format(name=self.path.name))
    #     for xml in self.files:
    #         pbtxt.extend(XML(path=xml).pbtxt)
    #     return sorted(list(set(pbtxt)))


class Data(Source):
    # class Decorators:
    #     @classmethod
    #     def csv(cls, f):
    #         @wraps(f)
    #         def wrapper(self, *args, **kwargs):
    #             name, csv, cols = f(self, *args, **kwargs)
    #             path = os.path.join(self.path, '{name}.csv'.format(name=name))
    #             logger.info('Outputing csv to {csv}'.format(csv=path))
    #             return pandas.DataFrame(csv, columns=cols).to_csv(path)
    #
    #         return wrapper
    #
    #     @classmethod
    #     def pbtxt(cls, f):
    #         @wraps(f)
    #         def wrapper(self, *args, **kwargs):
    #             name, pbtxt = f(self, *args, **kwargs)
    #             path = os.path.join(self.path, '{name}.pbtxt'.format(name=name))
    #             logger.info('Outputing pbtxt to {pbtxt}'.format(pbtxt=path))
    #             with open(path, "w") as output:
    #                 output.write(pbtxt)
    #
    #         return wrapper

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Images.factory(path=Path(os.path.join(self.path, kwargs.get('images', Images.name))))
        Labels.factory(path=Path(os.path.join(self.path, kwargs.get('labels', Labels.name))))
        Train.factory(path=Path(os.path.join(self.path, kwargs.get('train', Train.name))))
        Test.factory(path=Path(os.path.join(self.path, kwargs.get('test', Test.name))))

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

    # @Decorators.csv
    # def _csv(self, *args):
    #     return args
    #
    # def csv(self):
    #     self._csv(Train.instance.name, *Train.instance.csv)
    #     self._csv(Test.instance.name, *Test.instance.csv)

    # @Decorators.pbtxt
    # def pbtxt(self):
    #     train, test, pbtxt = Train.instance.pbtxt, Test.instance.pbtxt, ""
    #     for i, c in enumerate(sorted(list(set(train + test)))):
    #         pbtxt = (pbtxt + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, c))
    #     return Labels.instance.name, pbtxt


class Images(Source):
    name = 'images'

    def split(self):
        return


class Labels(Source):
    name = 'labels'


class Train(Source):
    name = 'train'


class Test(Source):
    name = 'test'


class TFRecord:

    def __init__(self, **kwargs):
        self.xml = kwargs.get('xml')
        assert isinstance(self.xml, XML)
        self.path = os.path.join(Data.instance.path, '{name}.record'.format(name=self.xml.path.parent.name))
        self.writer = tf.io.TFRecordWriter(self.path)

    # def _writer(self):
    #     path = os.path.join(Data.instance.path, '{name}.record'.format(name=self.xml.path.parent.name))
    #     return tf.io.TFRecordWriter(path)

    @property
    def filename(self):
        return {Root.RootSchema.FILENAME: dataset_util.bytes_feature(self.xml.tree.root.filename.encode('utf8'))}

    @property
    def size(self):
        return {Size.SizeSchema.WIDTH: dataset_util.int64_feature(self.xml.tree.root.size.width),
                Size.SizeSchema.HEIGHT: dataset_util.int64_feature(self.xml.tree.root.size.height)}

    @property
    def object(self):
        name, label, xmin, ymin, xmax, ymax = [], [], [], [], [], []
        for o in self.xml.tree.root.objects:
            name.append(o.name.encode('utf8'))
            label.append(1)
            xmin.append(o.bndbox.xmin)
            ymin.append(o.bndbox.ymin)
            xmax.append(o.bndbox.xmax)
            ymax.append(o.bndbox.ymax)
        return {Object.ObjectSchema.NAME: dataset_util.bytes_list_feature(name),
                "label": dataset_util.int64_list_feature(label),
                BNDBOX.BNDBOXSchema.XMIN: dataset_util.int64_list_feature(xmin),
                BNDBOX.BNDBOXSchema.YMIN: dataset_util.int64_list_feature(ymin),
                BNDBOX.BNDBOXSchema.XMAX: dataset_util.int64_list_feature(xmax),
                BNDBOX.BNDBOXSchema.YMAX: dataset_util.int64_list_feature(ymax)}

    @property
    def tfrecord(self):
        return {**self.filename, **self.size, **self.object}

    def write(self):
        logger.info('Writing tfrecord to {path}'.format(path=self.path))
        self.writer.write(tf.train.Example(features=tf.train.Features(feature=self.tfrecord)).SerializeToString())


class CSV:
    pass


class XML:
    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        assert os.path.isfile(self.path)
        logger.info('Creating XML object from {name}'.format(name=self.path.name))
        self.tree = Tree(tree=ElementTree.parse(self.path))

    @property
    def csv(self):
        csv = []
        for o in self.tree.root.objects:
            csv.append((self.tree.root.filename, self.tree.root.size.width, self.tree.root.size.height,
                        o.name, o.bndbox.xmin, o.bndbox.ymin, o.bndbox.xmax, o.bndbox.ymax))
        return csv, [Root.RootSchema.FILENAME, *Size.SizeSchema.values(), Object.ObjectSchema.NAME,
                     *BNDBOX.BNDBOXSchema.values()]

    @property
    def python(self):
        return dataset_util.recursive_parse_xml_to_dict(self.tree.root.root)

    @property
    def tfrecord(self):
        return TFRecord(xml=self)


class Tree:
    def __init__(self, **kwargs):
        self.tree = kwargs.get('tree')
        self.root = Root(root=self.tree.getroot())


class Root:
    class RootSchema(Schema):
        FILENAME = 'filename'
        SIZE = 'size'
        OBJECT = 'object'

    def __init__(self, **kwargs):
        self.root = kwargs.get('root')
        self.filename = self.root.find(self.RootSchema.FILENAME).text
        self.size = Size(size=self.root.find(self.RootSchema.SIZE))
        self.objects = [Object(object=o) for o in self.root.findall(self.RootSchema.OBJECT)]


class Size:
    class SizeSchema(Schema):
        WIDTH = 'width'
        HEIGHT = 'height'

    def __init__(self, **kwargs):
        self.size = kwargs.get('size')
        self.width = int(self.size.find(self.SizeSchema.WIDTH).text)
        self.height = int(self.size.find(self.SizeSchema.HEIGHT).text)


class Object:
    class ObjectSchema(Schema):
        NAME = 'name'
        BNDBOX = 'bndbox'

    def __init__(self, **kwargs):
        self.object = kwargs.get('object')
        self.name = self.object.find(self.ObjectSchema.NAME).text
        self.bndbox = BNDBOX(bndbox=self.object.find(self.ObjectSchema.BNDBOX))


class BNDBOX:
    class BNDBOXSchema(Schema):
        XMIN = 'xmin'
        YMIN = 'ymin'
        XMAX = 'xmax'
        YMAX = 'ymax'

    def __init__(self, **kwargs):
        self.bndbox = kwargs.get('bndbox')
        self.xmin = int(self.bndbox.find(self.BNDBOXSchema.XMIN).text)
        self.ymin = int(self.bndbox.find(self.BNDBOXSchema.YMIN).text)
        self.xmax = int(self.bndbox.find(self.BNDBOXSchema.XMAX).text)
        self.ymax = int(self.bndbox.find(self.BNDBOXSchema.YMAX).text)
