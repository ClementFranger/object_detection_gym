import numpy as np
import tensorflow as tf
from PIL import Image
from workout.utils import Source


class DetectionModelInterface(Source):
    name = 'saved_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        DetectionModel.factory(path=self.path)

    @property
    def detection_model(self):
        return DetectionModel.instance


class DetectionModel(Source):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('path is %s ' % self.path)
        self.model = tf.saved_model.load(str(self.path), tags=None, options=None)
        self.model = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def detect(self, **kwargs):
        input_tensor = kwargs.get('input_tensor')
        assert isinstance(input_tensor, ImageTensor)
        # tensor = input_tensor.tensor
        # print(tensor)
        # detections = self.model(tensor)
        print(input_tensor.tensor)
        return self.model(input_tensor.tensor)


class ImageTensor:
    def __init__(self, **kwargs):
        image = kwargs.get('image')
        if isinstance(image, str):
            print('image open')
        self.image = Image.open((image)) if isinstance(image, str) else image

    @property
    def tensor(self):
        print('convert to tensor')
        # image_np = np.array(Image.open((r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\71.jpg')))

        # # Things to try:
        # # Flip horizontally
        # # image_np = np.fliplr(image_np).copy()
        #
        # # Convert image to grayscale
        # # image_np = np.tile(
        # #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
        #
        # # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # input_tensor = tf.convert_to_tensor(image_np)
        # # The model expects a batch of images, so add an axis with `tf.newaxis`.
        # input_tensor = input_tensor[tf.newaxis, ...]
        # tensor =
        # tensor =
        tensor = tf.convert_to_tensor(self.np_image)
        tensor = tensor[tf.newaxis, ...]
        return tf.convert_to_tensor(self.np_image)

    @property
    def np_image(self):
        return np.array(self.image)
