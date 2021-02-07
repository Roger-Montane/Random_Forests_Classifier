import numpy as np
import abc


class DecisionTree(metaclass=abc.ABCMeta):
    """
    An interface used to represent a Decision Tree.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    predicts(x):
        Abstract method for making a prediction.
    """

    @abc.abstractmethod
    def predict(self, x):
        """
        Abstract method for making a prediction

        ...

        Attributes
        ----------
        x : Dataset

        Returns
        -------
        NotImplementedError
        """
        raise NotImplementedError
