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

    def __init__(self, **kwargs):
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

    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def files(self):
        files = list(self.path.glob('*.xml'))
        logger.info('Found {count} xml files in {name}'.format(count=len(files), name=self.path.name))
        return files

    @property
    def xml(self):
        xml = []
        logger.info('Retrieving {name} data as xml'.format(name=self.path.name))
        for x in self.files:
            xml.extend(LabelIMGXML(path=x).xml)
        return xml

    @property
    def csv(self):
        csv = []
        logger.info('Retrieving {name} data as csv'.format(name=self.path.name))
        for xml in self.files:
            csv.extend(LabelIMGXML(path=xml).csv)
        return csv, LabelIMGXML.LabelIMGSchema.values()

    @property
    def pbtxt(self):
        pbtxt = []
        logger.info('Retrieving {name} data as pbtxt'.format(name=self.path.name))
        for xml in self.files:
            pbtxt.extend(LabelIMGXML(path=xml).pbtxt)
        return sorted(list(set(pbtxt)))


class Data(Source):

    class Decorators:
        @classmethod
        def csv(cls, f):
            @wraps(f)
            def wrapper(self, *args, **kwargs):
                name, csv, cols = f(self, *args, **kwargs)
                path = os.path.join(self.path, '{name}.csv'.format(name=name))
                logger.info('Outputing csv to {csv}'.format(csv=path))
                return pandas.DataFrame(csv, columns=cols).to_csv(path)
            return wrapper

        @classmethod
        def pbtxt(cls, f):
            @wraps(f)
            def wrapper(self, *args, **kwargs):
                name, pbtxt = f(self, *args, **kwargs)
                path = os.path.join(self.path, '{name}.pbtxt'.format(name=name))
                logger.info('Outputing pbtxt to {pbtxt}'.format(pbtxt=path))
                with open(path, "w") as output:
                    output.write(pbtxt)
            return wrapper

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

    @Decorators.csv
    def _csv(self, *args):
        return args

    def csv(self):
        self._csv(Train.instance.name, *Train.instance.csv)
        self._csv(Test.instance.name, *Test.instance.csv)

    @Decorators.pbtxt
    def pbtxt(self):
        train, test, pbtxt = Train.instance.pbtxt, Test.instance.pbtxt, ""
        for i, c in enumerate(sorted(list(set(train + test)))):
            pbtxt = (pbtxt + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, c))
        return Labels.instance.name, pbtxt


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


class LabelIMGXML:
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

    def _pbtxt(self):
        pbtxt = []
        root = ElementTree.parse(self.path).getroot()
        for o in root.findall(self.XMLSchema.OBJECT):
            pbtxt.append(o.find(self.LabelIMGSchema.NAME).text)
        return pbtxt

    @property
    def csv(self):
        logger.info('Converting {name} to csv'.format(name=self.path.name))
        return self._csv()

    @property
    def pbtxt(self):
        logger.info('Converting {name} to pbtxt'.format(name=self.path.name))
        return self._pbtxt()


class CSV:
    pass


class XML:

    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        assert os.path.isfile(self.path)
        self.tree = Tree(tree=ElementTree.parse(self.path))


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
