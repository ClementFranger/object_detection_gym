import os


class LabelIMG:
    instance = None
    data = None

    @classmethod
    def factory(cls):
        if cls.instance is None:
            cls.instance = cls()
        assert isinstance(cls.instance, cls)
        return cls.instance

    @property
    def labels(self):
        return os.path.join(self.data, 'labels')

    @property
    def images(self):
        return os.path.join(self.data, 'images')

    @property
    def train(self):
        return os.path.join(self.data, 'train')

    @property
    def test(self):
        return os.path.join(self.data, 'test')

    def csv(self):
        return