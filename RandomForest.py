import numpy as np
from Dataset import Dataset
from SplitNode import SplitNode
from Leaf import Leaf
from Gini import Gini
from Entropy import Entropy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('randomforest.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class RandomForest():
    """
    A class used to generate a Random Forest.

    ...

    Attributes
    ---------
    max_depth : int
        The maximum depth of the trees
    min_size_split : int
        The inimum number of elements in a node to keep making childs
    ratio_samples : float
        Reduce the dataset for bagging [0,1]
    num_trees : int
        The number of trees
    num_features_node : int
        The number of different features in each node
    criterion_name : str
        The name of the criterion used

    Methods
    -------
    fit(X, y)
        Makes a trees from given Dataset.
    predict(x)
        Makes a prediction.
    """

    def __init__(self, max_depth, min_size_split, ratio_samples,
                 num_trees, num_features_node, criterion_name, split_feature_values=None):
        """
        Parameters
        ----------
        max_depth : int
            The maximum depth of the trees
        min_size_split : int
            The inimum number of elements in a node to keep making childs
        ratio_samples : float
            Reduce the dataset for bagging [0,1]
        num_trees : int
            The number of trees
        num_features_node : int
            The number of different features in each node
        criterion_name : str
            The name of the criterion used
        """

        self._max_depth = max_depth
        self._min_size_split = min_size_split
        self._ratio_samples = ratio_samples
        self._num_trees = num_trees
        self._num_features_node = num_features_node
        self._criterion_name = criterion_name
        if (criterion_name == "Gini"):
            self._criterion = Gini()
            logger.info("Criterion set to {}".format(criterion_name))
        elif (criterion_name == "Entropy"):
            self._criterion = Entropy()
            logger.info("Criterion set to {}".format(criterion_name))
        else:
            logger.error(NotImplementedError, "when chosing a criterion.")
        self._decisionTrees = []  # llista d'arbres
        self.values = split_feature_values

    def fit(self, X, y):
        """
        TODO Makes a tree from a Dataset??.

        Parameters
        ----------
        X : numpy.ndarray
            Matrix with samples
        y : numpy.ndarray
            Matrix with classes

        Returns
        -------
        None
        """

        try:
            dataset = Dataset(X, y)
            self._decisionTrees = self._make_trees(dataset)
            logger.info("Fit done.")
        except Exception as e:
            logger.error(
                "Something failed during fitting:\{}".format(str(e)))

    def predict(self, x):
        """
        Makes a prediction based on all trees results.

        Parameters
        ----------
        x : numpy.ndarray
            Matrix with samples

        Returns
        -------
            numpy.ndarray
            Matrix with predictions
        """

        try:
            ypred = []
            for element in x:
                predictions = [tree.predict(element)
                               for tree in self._decisionTrees]
                ypred.append(self._combine_predictions(predictions))
                logger.info("A prediction has been made.")

        except Exception as e:
            logger.error(
                "Something went wrong when making a prediction:\n{}".format(str(e)))

        else:
            return np.array(ypred, dtype=np.int)

    def _combine_predictions(self, predictions):
        try:
            logger.info("Predictions combined.")
            return max(set(predictions), key=predictions.count)

        except Exception as e:
            logger.error(
                "Something falied when combining predictions:\n{}".format(str(e)))

    def _make_trees(self, dataset):
        try:
            trees = []
            for i in range(self._num_trees):
                subset = dataset.subset_from_ratio(self._ratio_samples)
                logger.info("Tree number {}:\n    subset's X = {}\n    subset's y = {}".format(i, subset.get_X(), subset.get_y()))
                trees.append(self._build_tree(subset))
                logger.info("A tree has been made.")

        except Exception as e:
            logger.error(
                "While making trees something happened:\n{}".format(str(e)))

        else:
            return trees

    def _build_tree(self, dataset):
        try:
            depth = 1
            root = self._make_split_node(dataset, depth)
            logger.info("A tree has been builded.")

        except Exception as e:
            logger.error(
                "Something went wrong building a tree:\n{}".format(str(e)))

        else:
            return root

    def _make_split_node(self, dataset, depth):
        try:
            features = []

            logger.info("----------------DEPTH = {}----------------".format(depth))

            while len(features) < self._num_features_node:
                index = np.random.randint(low=0, high=dataset.num_features)
                if index not in features:
                    features.append(index)

            best_feature, best_value, best_score, best_split = np.Inf, np.Inf, np.Inf, None

            for feature in features:
                #-------------------------------Iris i Sonar--------------------------------
                #self.values = list(set(element[feature] for element in dataset.get_X()))
                #---------------------------------------------------------------------------
                for value in self.values:
                    left_dataset, right_dataset = self._compute_split(
                        feature, value, dataset)

                    if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
                        continue

                    score = self._criterion.evaluate_criterion(
                        left_dataset, right_dataset)
                    if score < best_score:
                        best_feature, best_value, best_score, best_split = feature, value, score, [
                            left_dataset, right_dataset]
            logger.debug("left dataset = {}, right dataset = {}".format(best_split[0], best_split[1]))
            node = SplitNode(best_feature, best_value)
            logger.info("Splitnode created")
            self._make_childs(node, depth + 1, best_split)

        except Exception as e:
            logger.error(
                "Node couldn't make a split due to:\n{}".format(str(e)))

        else:
            return node

    def _compute_split(self, index, value, dataset):
        try:
            index_left = dataset.get_X()[:, index] < value
            index_right = dataset.get_X()[:, index] >= value
            left = dataset.subset_from_indices(index_left)
            right = dataset.subset_from_indices(index_right)
            logger.debug("Split computed.")

        except Exception as e:
            logger.error(
                "Computing a split went wrong due to:\n{}".format(str(e)))

        else:
            return left, right

    def _make_childs(self, node, depth, better_split):
        try:
            logger.debug("better_split = ", better_split)
            # Esquerra
            if (depth >= self._max_depth or better_split[0].num_samples < self._min_size_split):
                node.left = self._make_leaf(better_split[0])
                logger.info("A leaf has been created.")
            else:
                node.left = self._make_split_node(better_split[0], depth)

            # Dreta
            if (depth >= self._max_depth or better_split[1].num_samples < self._min_size_split):
                node.right = self._make_leaf(better_split[1])
                logger.info("A leaf has been created.")
            else:
                node.right = self._make_split_node(better_split[1], depth)

        except Exception as e:
            logger.error(
                "Something failed while making childs:\n{}".format(str(e)))

    def _make_leaf(self, dataset):
        try:
            return Leaf(dataset)

        except Exception as e:
            logger.error(
                "Something failed while making a leaf:\n{}".format(str(e)))
