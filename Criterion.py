import abc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('client.log')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
# stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Criterion(metaclass=abc.ABCMeta):
    """
    An interface to make a Strategy pattern fro different cirterions.
    
    ...

    Attributes
    ----------
    None

    Methods
    -------
    evaluate_criterion():
        Abstract method for evaluating a criterion.
    """

    @abc.abstractmethod
    def evaluate_criterion(self):
        """
        Abstract method for evaluating a criterion.

        ...

        Attributes
        ----------
        None

        Returns
        -------
        NotImplementedError
        """
        raise NotImplementedError
