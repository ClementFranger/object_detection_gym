import os
import logging
from pathlib import Path
from xml.etree import ElementTree
from workout.utils import Schema
from object_detection.utils import dataset_util


logger = logging.getLogger(__name__)


class XML:
    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        assert os.path.isfile(self.path)
        logger.info('Creating XML object from {name}'.format(name=self.path.name))
        self.tree = Tree(tree=ElementTree.parse(self.path))
        self.image = kwargs.get('image')

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


class TFRecord:
    def __init__(self, **kwargs):
        self.xml = kwargs.get('xml')
        assert isinstance(self.xml, XML)

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
    def image(self):
        return {'image/encoded': dataset_util.bytes_feature(self.xml.image.encoded),
                'image/format': dataset_util.bytes_feature('jpg'.encode('utf8'))}

    @property
    def tfrecord(self):
        return {**self.filename, **self.size, **self.object, **self.image}
