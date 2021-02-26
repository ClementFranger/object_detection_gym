import os
import logging
from pathlib import Path
from xml.etree import ElementTree

from workout.labelimg.images import Images
from workout.utils import Schema, Image
from object_detection.utils import dataset_util


logger = logging.getLogger(__name__)


class XML:
    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        assert os.path.isfile(self.path)
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

    @property
    def image(self):
        image = os.path.join(Images.instance.path, self.path.name.replace('xml', 'jpg'))
        assert os.path.isfile(image)
        return Image(path=image)


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
        return {'image/filename': dataset_util.bytes_feature(self.xml.tree.root.filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(self.xml.tree.root.filename.encode('utf8'))}

    @property
    def size(self):
        return {'image/width': dataset_util.int64_feature(self.xml.tree.root.size.width),
                'image/height': dataset_util.int64_feature(self.xml.tree.root.size.height)}

    @property
    def object(self):
        name, label, xmin, ymin, xmax, ymax = [], [], [], [], [], []
        for o in self.xml.tree.root.objects:
            name.append(o.name.encode('utf8'))
            label.append(1)
            xmin.append(o.bndbox.xmin / self.xml.tree.root.size.width)
            ymin.append(o.bndbox.ymin / self.xml.tree.root.size.width)
            xmax.append(o.bndbox.xmax / self.xml.tree.root.size.height)
            ymax.append(o.bndbox.ymax / self.xml.tree.root.size.height)
        return {'image/object/class/text': dataset_util.bytes_list_feature(name),
                'image/object/class/label': dataset_util.int64_list_feature(label),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax)}

    @property
    def image(self):
        return {'image/encoded': dataset_util.bytes_feature(self.xml.image.encoded),
                'image/format': dataset_util.bytes_feature('jpg'.encode('utf8'))}

    @property
    def tfrecord(self):
        return {**self.filename, **self.size, **self.object, **self.image}
