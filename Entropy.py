import numpy as np
from Criterion import Criterion
from Dataset import Dataset
import sklearn.datasets


class Entropy(Criterion):
    """
    A class to evaluate with the Entropy Criterion.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    evaluate_criterion(left_dataset, right_dataset)
        Evaluates a Dataset with the Entropy Criterion.
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        pass

    def evaluate_criterion(self, dataset):
        """
        Evaluates a Dataset with the Entropy Criterion.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        float
            The value of Entropy after evaluation.
        """
        n = len(self.dataset)
        return (-1) * self._sumatory(self.dataset)

    def _sumatory(self, dataset):
        sumatory = 0
        for i, c in enumerate(dataset):
            _, class_frequency = np.unique(c, return_counts=True)
            sumatory += class_frequency * np.log(class_frequency)
        return sumatory