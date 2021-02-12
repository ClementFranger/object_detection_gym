import os

from workout.labelimg import LabelIMG


class DofusLabelIMG(LabelIMG):
    data = os.path.join(os.path.dirname(__file__), 'data')
