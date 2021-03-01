import logging
import os

import cv2

import PIL
import tensorflow as tf
from object_detection.utils import label_map_util

from workout.record.record import Record
from workout.vision.detection import Detection
from workout.vision.model import Model
from workout.vision.tensor import ImageTensor
from workout.utils import Image, MSSImage

logger = logging.getLogger(__name__)


# TODO : solve this session problem so that we can load model from another class. For now do everything here
class Vision:
    instance = None

    def __init__(self, **kwargs):
        Model.factory(**kwargs)
        Record.factory(**kwargs)

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def model(self):
        return Model.instance

    @property
    def record(self):
        return Record.instance

    def model_with_signatures(self):
        """ important to set signature in the run function """
        return self.model.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def category_index(self, **kwargs):
        return label_map_util.create_category_index_from_labelmap(kwargs.get('labels'), use_display_name=True)

    def detect_image(self, **kwargs):
        category_index = self.category_index(**kwargs)
        model, image = self.model_with_signatures(), Image(path=kwargs.get('image'))
        detections = Detection(model=model, tensor=ImageTensor(image=image).tensor, **kwargs)
        img = image.draw_boxes(detections=detections, category_index=category_index)

        im = PIL.Image.fromarray(img)
        im.save("your_file.jpeg")

    def detect_video(self, **kwargs):
        category_index, model = self.category_index(**kwargs), self.model_with_signatures()
        # model, image = self.model_with_signatures(), Image(path=kwargs.get('image'))
        # detections = Detection(model=model, tensor=ImageTensor(image=image).tensor, **kwargs)
        # img = image.draw_boxes(detections=detections, category_index=category_index)
        # Record.instance.record_video_with_detection(category_index=category_index, model=model, **kwargs)
        from workout.record.application import Application
        Application.factory(**kwargs)
        from workout.writer.record import RecordWriter
        writer = RecordWriter(path=os.path.join(self.record.path, kwargs.get('output')), application=self.record.application,
                              **kwargs).writer

        logger.info('Writting recorded video into {path}'.format(path=self.record.path))
        import mss
        with mss.mss() as sct:
            while True:
                import time
                start_time = time.time()
                image = MSSImage(image=sct.grab(self.record.application.monitor))

                """ detection """
                detections = Detection(tensor=ImageTensor(image=image.cleaneda).tensor, model=model, **kwargs)
                image = image.draw_boxes(detections=detections, category_index=category_index, **kwargs)

                writer.write(image)
                import keyboard
                if keyboard.is_pressed('a'):
                    break

                logger.info('Process time is {time}'.format(time=time.time() - start_time))
        logger.info('Successfully wrote recorded video into {path}'.format(path=self.record.path))

        writer.release()
        cv2.destroyAllWindows()
    # def detect_video(self, **kwargs):
    #     # Application.factory(**kwargs)
    #     Record.factory(**kwargs)
    #     out = cv2.VideoWriter('detection_video.avi', cv2.VideoWriter_fourcc(*"XVID"), 60.0, Record.instance.application.size)
    #     print(int((1/int(60))*1000))
    #     import mss
    #     with mss.mss() as sct:
    #         while True:
    #             img = Record.instance.grab(sct=sct)
    #             frame = Record.instance.color(image=img)
    #
    #             out.write(frame)
    #             # category_index = self.category_index(**kwargs)
    #             # model, image = self.model_with_signatures(), Image(path=kwargs.get('image'))
    #             # detections = Detection(model=model, tensor=ImageTensor(image=image).tensor, **kwargs)
    #             # img = image.draw_boxes(detections=detections, category_index=category_index)
    #             #
    #             # im = PIL.Image.fromarray(img)
    #             # im.save("your_file.jpeg")
    #             if cv2.waitKey(16) == ord('q'):
    #                 break
    #     out.release()
    #     cv2.destroyAllWindows()