from tensorflow.python.saved_model.load import _WrapperFunction

from test_workout import TestTensorflow
from workout.model.trained_model import TrainedModel


class TestTrainedModel(TestTensorflow):

    def setUp(self):
        super().setUp()
        self.trained_model = TrainedModel.factory(source=self.graph)

    def test_model(self):
        assert isinstance(TrainedModel.instance.model_with_signatures, _WrapperFunction)
