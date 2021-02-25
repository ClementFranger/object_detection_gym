import logging

from workout.labelimg.tfrecord import XML
from workout.utils import Source
from workout.writer.tfrecord import TFRecordWriter

logger = logging.getLogger(__name__)


class Test(Source):
    """ directory containing labelimg testing data """
    name = 'test'

    @property
    def tfrecords(self):
        tfrecords = []
        for x in list(self.path.glob('*.xml')):
            tfrecords.append(XML(path=x).tfrecord)
        return tfrecords

    def tfrecords_write(self, **kwargs):
        writer = TFRecordWriter(path=kwargs.get('path'))
        logger.info('Writting testing data into {path}'.format(path=self.path))
        for r in self.tfrecords:
            writer.write(tfrecord=r.tfrecord)
        logger.info('Successfully wrote testing data into {path}'.format(path=self.path))
