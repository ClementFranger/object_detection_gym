
def get_image_resizer_config(model_config):
  """Returns the image resizer config from a model config.
  Args:
    model_config: A model_pb2.DetectionModel.
  Returns:
    An image_resizer_pb2.ImageResizer.
  Raises:
    ValueError: If the model type is not recognized.
  """
  meta_architecture = model_config.WhichOneof("model")
  if meta_architecture == "faster_rcnn":
    return model_config.faster_rcnn.image_resizer
  if meta_architecture == "ssd":
    return model_config.ssd.image_resizer

  raise ValueError("Unknown model type: {}".format(meta_architecture))


import unittest


class TestXML(unittest.TestCase):
    config = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\pipeline.config'

    def test_(self):
        print(get_image_resizer_config(self.config))