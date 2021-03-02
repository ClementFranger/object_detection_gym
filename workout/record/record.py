import os
import logging
import keyboard
import time
import cv2
import mss

from workout.image import MSSImage
from workout.record.application import Application
from workout.utils import Source
from workout.vision.detection import Detection
from workout.writer.record import RecordWriter

logger = logging.getLogger(__name__)


class Record(Source):
    """ directory containing all recorded data """
    name = 'record'

    @property
    def application(self):
        return Application.instance

    def record_video(self, **kwargs):
        Application.factory(**kwargs)
        writer = RecordWriter(path=os.path.join(self.path, kwargs.get('output')), application=self.application, **kwargs).writer

        logger.info('Writting recorded video into {path}'.format(path=self.path))
        with mss.mss() as sct:
            while True:
                start_time = time.time()
                image = MSSImage(image=sct.grab(self.application.monitor))
                writer.write(image.cleaned)

                if keyboard.is_pressed('x'):
                    break

                logger.info('Process time is {time}'.format(time=time.time() - start_time))
        logger.info('Successfully wrote recorded video into {path}'.format(path=self.path))

        writer.release()
        cv2.destroyAllWindows()

    def record_video_with_detection(self, **kwargs):
        Application.factory(**kwargs)
        writer = RecordWriter(path=os.path.join(self.path, kwargs.get('output')), application=self.application,
                              **kwargs).writer

        logger.info('Writting recorded video into {path}'.format(path=self.path))
        with mss.mss() as sct:
            while True:
                start_time = time.time()
                image = MSSImage(image=sct.grab(self.application.monitor))

                """ detection """
                detections = Detection(tensor=image.tensor, **kwargs)
                image = image.draw_boxes(detections=detections, **kwargs)

                writer.write(image)
                if keyboard.is_pressed('x'):
                    break

                logger.info('Process time is {time}'.format(time=time.time() - start_time))
        logger.info('Successfully wrote recorded video into {path}'.format(path=self.path))

        writer.release()
        cv2.destroyAllWindows()
