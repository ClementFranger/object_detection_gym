import logging
from pyautogui import getWindowsWithTitle


logger = logging.getLogger(__name__)


class Application:
    instance = None

    def __init__(self, **kwargs):
        self.title = kwargs.get('title')
        self.window = self._window()
        self.maximize()

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance

    def _window(self):
        window = getWindowsWithTitle(self.title)
        logger.info('Found {count} window for {title}'.format(count=len(window), title=self.title))
        return next(iter(window), None)

    @property
    def size(self):
        return self.window.size

    @property
    def topleft(self):
        return self.window.topleft

    @property
    def bottomright(self):
        return self.window.bottomright

    @property
    def monitor(self):
        return {"top": self.topleft.x, "left": self.topleft.y, "width": self.size.width, "height": self.size.height}

    def maximize(self):
        self.window.maximize()
