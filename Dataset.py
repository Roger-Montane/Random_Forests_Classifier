import numpy as np
from random import randrange
import sklearn.datasets
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('dataset.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Dataset():
    """
    A class used to represent a Dataset.

    ...

    Attributes
    ----------
    X : numpy.ndarray
        Matrix with samples
    y : numpy.ndarray
        Matrix with classes

    Methods
    -------
    get_X()
        Gets X.
    get_y()
        Gets y.
    distribution()
        Returns the distribution of all classes.
    subset_from_indices()
        Returns a part of the Dataset, determined by the input array.
    subset_from_ratio(ratio)
        Returns a subset of size determined by the input ratio.
    """

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray
            Matrix with samples
        y : numpy.ndarray
            Matrix with classes
        """

        self._X = X
        self._y = y
        self.num_features = X.shape[1]
        self.num_samples = X.shape[0]
        self._labels = np.unique(self._y)

    def get_X(self):
        """
        Gets X.

        Parameters
        ----------
        None

        Returns
        -------
        self._X : numpy.ndarray
            Matrix with samples

        """

        return self._X

    def get_y(self):
        """
        Gets y.

        Parameters
        ----------
        None

        Returns
        -------
        self._y : numpy.ndarray
            Matrix with classes

        """

        return self._y

    def distribution(self):
        """
        Returns the distribution of all classes.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Array containing the distribution of each class.
        """

        counts = np.array([list(self._y).count(label)
                           for label in self._labels])
        return counts / float(self.num_samples)

    def subset_from_indices(self, idx):
        """
        Returns a part of the Dataset, determined by the input array.

        Parameters
        ----------
        idx : numpy.ndarray of booleans
            Matrix of booleans.

        Returns
        -------
        Dataset
            Subset constructed by the True indices from the input array.

        """

        return Dataset(self._X[idx], self._y[idx])

    def subset_from_ratio(self, ratio):
        """
        Returns a subset of size determined by the input ratio.

        Parameters
        ----------
        ratio : float
            Ratio of samples to be used.

        Returns
        -------
        Dataset
            Subset of of the original set with size ratio times original size.

        """

        try:
            subset_size = int(self.num_samples * ratio)
            idx = np.random.permutation(range(self.num_samples))[0:subset_size]
            logger.debug("Samples reduced for training.")

        except Exception as e:
            logger.error(
                "Something failed reducing the Dataset:\n{}".format(str(e)))

        else:
            return Dataset(self._X[idx], self._y[idx])
