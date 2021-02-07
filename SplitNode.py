import numpy as np
from DecisionTree import DecisionTree
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('splitnode.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class SplitNode(DecisionTree):
    """
    A class to represent a binari split of a tree.

    ...

    Attributes
    ----------
    feature : int
        The value of the index of a certain feature.
    value : int
        The value of array[feature] where array is the y of the Dataset.

    Methods
    -------
    predicts(x)
        Makes a prediction.
    """

    def __init__(self, feature, value):
        """
        Parameters
        ----------
        feature : int
            The value of the index of a certain feature.
        value : int
            The value of array[feature] where array is the y of the Dataset.
        """

        self._feature = feature
        self._value = value
        self.left = None
        self.right = None

    def predict(self, X):
        """
        Makes a prediction.

        ...

        Parameters
        ----------
        X : np.ndarray
            Reduced set of data that has arrived to that node

        Returns
        -------
        int
            Class prediction for values
        """
        try:
            if X[self._feature] < self._value:
                logger.debug("Left prediction made on {}".format(__name__))
                return self.left.predict(X)

            logger.debug("Right prediction made on {}".format(__name__))
            return self.right.predict(X)

        except Exception as e:
            logger.error(
                "Something failed predicting on a {}:\n{}".format(__name__, str(e)))
