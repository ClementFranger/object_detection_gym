# import os
# import logging
# import time
# import cv2
# from pyautogui import getWindowsWithTitle
# import numpy as np
#
# from workout.utils import Source
#
# logger = logging.getLogger(__name__)
#
#
# class Video(Source):
#     name = 'record'
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         Application.factory(**kwargs)
#
#     def _resize(self, **kwargs):
#         return cv2.resize(kwargs.get('image'), Application.instance.size)
#
#     def grab(self, **kwargs):
#         return self._resize(image=np.array(kwargs.get('sct').grab(Application.instance.monitor)))
#
#     def color(self, **kwargs):
#         frame = cv2.cvtColor(kwargs.get('image'), cv2.COLOR_BGR2RGB)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         return frame
#
#     def record(self):
#         import mss
#         count = 0
#         with mss.mss() as sct:
#             while True:
#                 count = count + 1
#                 # last_time = time.time()
#                 img = self.grab(sct=sct)
#                 frame = self.color(image=img)
#
#                 # cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
#                 #             (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                 logger.info('Saving file {name}'.format(name=count))
#                 cv2.imwrite(os.path.join(self.path, '{count}.jpg'.format(count=count)), frame)
#                 time.sleep(0.5)
#
#                 if cv2.waitKey(25) & 0xFF == ord("q"):
#                     break
#
#
# class Application:
#     instance = None
#
#     def __init__(self, **kwargs):
#         self.title = kwargs.get('title')
#         self.window = self._window()
#         self.maximize()
#
#     @classmethod
#     def factory(cls, **kwargs):
#         if cls.instance is None:
#             cls.instance = cls(**kwargs)
#         assert isinstance(cls.instance, cls)
#         return cls.instance
#
#     def _window(self):
#         window = getWindowsWithTitle(self.title)
#         logger.info('Found {count} window for {title}'.format(count=len(window), title=self.title))
#         return next(iter(window), None)
#
#     @property
#     def size(self):
#         return self.window.size
#
#     @property
#     def topleft(self):
#         return self.window.topleft
#
#     @property
#     def bottomright(self):
#         return self.window.bottomright
#
#     @property
#     def monitor(self):
#         return {"top": self.topleft.x, "left": self.topleft.y, "width": self.size.width, "height": self.size.height}
#
#     def maximize(self):
#         logger.info('Maximizing {window}'.format(window=self.title))
#         self.window.maximize()
