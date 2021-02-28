import PIL
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils

from workout.detection.model import Model
from workout.detection.tensor import ImageTensor
from workout.utils import Source, Image


# TODO : solve this session problem so that we can load model from another class. For now do everything here
class Detection(Source):
    name = 'saved_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Model.factory(path=self.path)

    @property
    def model(self):
        return Model.instance

    def _signatures(self, graph):
        return graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def detect_image(self, **kwargs):
        category_index = label_map_util.create_category_index_from_labelmap(kwargs.get('labels'), use_display_name=True)
        """ do not overwrite previous variable """
        graph = tf.saved_model.load(str(self.model.path), tags=None, options=None)
        # infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        infer = self._signatures(graph)
        image = Image(path=kwargs.get('image'))
        input_tensor = ImageTensor(image=image)
        detections = infer(input_tensor.tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        img = visualization_utils.visualize_boxes_and_labels_on_image_array(
            image.np_array,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        im = PIL.Image.fromarray(img)
        im.save("your_file.jpeg")
        return

    def detect_image2(self, **kwargs):
        import numpy as np
        from PIL import Image
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils

        category_index = label_map_util.create_category_index_from_labelmap(kwargs.get('labels'), use_display_name=True)
        graph = tf.saved_model.load(str(self.model.path), tags=None, options=None)
        infer = graph.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        image_np = np.array(Image.open((r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\overwatch\data\images\1614284494123.jpg')))

        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = infer(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        img = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        im = Image.fromarray(img)
        im.save("your_file.jpeg")


    def detect_video(self):
        return



    # def detect(self, **kwargs):
    #     input_tensor = kwargs.get('input_tensor')
    #     assert isinstance(input_tensor, ImageTensor)
    #     # tensor = input_tensor.tensor
    #     # print(tensor)
    #     # detections = self.model(tensor)
    #     print(input_tensor.tensor)
    #     return self.model(input_tensor.tensor)



