import numpy as np
from Criterion import Criterion
from Dataset import Dataset
import sklearn.datasets
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('gini.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Gini(Criterion):
    """
    A class to evaluate with the Gini Criterion.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    evaluate_criterion(left_dataset, right_dataset)
        Evaluates a Dataset with the Gini Criterion.
    """

    def __init__(self):
        """
        Parameters
        ----------
        None
        """

        pass

    def evaluate_criterion(self, left_dataset, right_dataset):
        """
        Evaluates a Dataset with the Gini Criterion.

        Parameters
        ----------
        left_dataset : Dataset
            Dataset partition according to a certain condition.

        right_dataset : Dataset
            Dataset partition according to a certain condition.

        Returns
        -------
        gini : float
            The value of gini after evaluation
        """
        try:
            n_instances = left_dataset.num_samples + right_dataset.num_samples
            gini = 0.0
            for dataset in [left_dataset, right_dataset]:
                size = dataset.num_samples
                if size == 0:
                    continue
                probs = dataset.distribution()
                score = np.sum(probs**2)
                gini += (1.0 - score) * (size / float(n_instances))
                logger.debug("Dataset evaluated with {}".format(__name__))

        except Exception as e:
            logger.error(
                "Something failed evaluating with {} criterion:\n{}".format(__name__, str(e)))

        else:
            return gini
