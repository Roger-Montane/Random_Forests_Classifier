import numpy as np
from DecisionTree import DecisionTree
from scipy import stats as s
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('leaf.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Leaf(DecisionTree):
    """
    A class to represent a Leaf of a tree.

    ...

    Attributes
    ----------
    outcome : Dataset
        Classified values.

    Methods
    -------
    predict(x)
        Makes a prediction.
    """

    def __init__(self, outcome):
        """
        Parameters
        ----------
        outcome : Dataset
            Classified values.
        """

        self._outcome = outcome

    def predict(self, x):
        """
        Predicts the outcome of a leaf.

        Parameters
        ----------
        x : numpy.ndarray
            Matrix with samples

        Returns
        -------
        int
            Class prediction for values, which will be the most common one (i.e. the mode).
        """
        try:
            logger.debug("Prediction made for a {}".format(__name__))
            return int(s.mode(self._outcome.get_y())[0])
        except Exception as e:
            logger.error(
                "Something failed predicting on a {}:\{}".format(__name__, str(e)))
