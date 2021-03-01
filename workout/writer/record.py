import logging
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class RecordWriter:
    def __init__(self, **kwargs):
        self.path = Path(kwargs.get('path'))
        self.application = kwargs.get('application')
        logger.info('Initializing VideoWriter to {path}'.format(path=self.path))
        """ make sure the fps match the computer processing speed """
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"XVID"), kwargs.get('fps', 30.0),
                                      self.application.size)

    def write(self, **kwargs):
        """ important : do not convert color before detection (idk why) """
        image = cv2.cvtColor(kwargs.get('image'), cv2.COLOR_BGR2RGB)
        self.writer.write(image)
