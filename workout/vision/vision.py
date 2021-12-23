# import logging
# import os
# import time
#
# import cv2
#
# import PIL
# import keyboard
# import mss
# import tensorflow as tf
# from object_detection.utils import label_map_util
#
# from workout.image import MSSImage, PathImage
# from workout.record.application import Application
# from workout.record.record import Record
# from workout.vision.detection import Detection
# from workout.vision.trained_model import TrainedModel
# from workout.writer.record import RecordWriter
#
# logger = logging.getLogger(__name__)
#
#
# class Vision:
#     instance = None
#
#     def __init__(self, **kwargs):
#         # TrainedModel.factory(**kwargs)
#         Record.factory(**kwargs)
#
#     @classmethod
#     def factory(cls, **kwargs):
#         if cls.instance is None:
#             cls.instance = cls(**kwargs)
#         assert isinstance(cls.instance, cls)
#         return cls.instance
#
#     # @property
#     # def trained_model(self):
#     #     return TrainedModel.instance
#
#     @property
#     def record(self):
#         return Record.instance
#
#     # def model_with_signatures(self):
#     #     """ important to set signature in the run function """
#     #     return self.trained_model.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#
#     def category_index(self, **kwargs):
#         return label_map_util.create_category_index_from_labelmap(kwargs.get('labels'), use_display_name=True)
#
#     def detect_image(self, **kwargs):
#         category_index = self.category_index(**kwargs)
#         model, image = self.model_with_signatures(), PathImage(path=kwargs.get('image'))
#         """ detection """
#         detections = Detection(image=image, model=model)
#         image = detections.draw_boxes(category_index=category_index, **kwargs)
#
#         im = PIL.Image.fromarray(image)
#         im.save("your_file.jpeg")
#
#     def detect_video(self, **kwargs):
#         category_index, model = self.category_index(**kwargs), self.model_with_signatures()
#         Application.factory(**kwargs)
#         record_writer = RecordWriter(path=os.path.join(self.record.path, kwargs.get('output')),
#                                      application=self.record.application, **kwargs)
#
#         logger.info('Writting recorded video into {path}'.format(path=self.record.path))
#         with mss.mss() as sct:
#             while True:
#                 start_time = time.time()
#                 # logger.info('Grabbing monitor %s' % self.record.application.monitor)
#                 image = MSSImage(image=sct.grab(self.record.application.monitor))
#                 # image = PathImage(path=kwargs.get('image'))
#                 # grab_time = time.time()
#                 # logger.info('grab time is %s' % (grab_time - start_time))
#
#                 """ detection """
#                 detections = Detection(image=image, model=model)
#                 image = detections.draw_boxes(category_index=category_index, **kwargs)
#
#                 write_time = time.time()
#                 # """ important : do not convert color before detection (idk why) """
#                 # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 # writer.write(image)
#                 record_writer.write(image=image)
#                 logger.info('write time is %s' % (time.time() - write_time))
#                 if keyboard.is_pressed('x'):
#                     break
#
#                 logger.info('Process time is {time}'.format(time=time.time() - start_time))
#         logger.info('Successfully wrote recorded video into {path}'.format(path=self.record.path))
#
#         record_writer.writer.release()
#         cv2.destroyAllWindows()
#     # def detect_video(self, **kwargs):
#     #     # Application.factory(**kwargs)
#     #     Record.factory(**kwargs)
#     #     out = cv2.VideoWriter('detection_video.avi', cv2.VideoWriter_fourcc(*"XVID"), 60.0, Record.instance.application.size)
#     #     print(int((1/int(60))*1000))
#     #     import mss
#     #     with mss.mss() as sct:
#     #         while True:
#     #             img = Record.instance.grab(sct=sct)
#     #             frame = Record.instance.color(image=img)
#     #
#     #             out.write(frame)
#     #             # category_index = self.category_index(**kwargs)
#     #             # model, image = self.model_with_signatures(), Image(path=kwargs.get('image'))
#     #             # detections = Detection(model=model, tensor=ImageTensor(image=image).tensor, **kwargs)
#     #             # img = image.draw_boxes(detections=detections, category_index=category_index)
#     #             #
#     #             # im = PIL.Image.fromarray(img)
#     #             # im.save("your_file.jpeg")
#     #             if cv2.waitKey(16) == ord('q'):
#     #                 break
#     #     out.release()
#     #     cv2.destroyAllWindows()
