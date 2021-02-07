import numpy as np
from RandomForest import RandomForest
import sklearn.datasets
from mnist import load
from sonar_figures import load_sonar
import timeit
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler('client.log')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def main():
    """ 
    # ----------------------Iris------------------------
    iris = sklearn.datasets.load_iris()
    print(iris.DESCR)
    X, y = iris.data, iris.target
    # --------------------------------------------------
    """

    """ # ----------------------Sonar------------------------
    X, y = load_sonar()
    print(X.shape, y.shape)
    # --------------------------------------------------- """


    """ # -------------------Iris i Sonar--------------------
    ratio_train_test = 0.8
    num_samples, num_features = X.shape
    idx = np.random.permutation(range(num_samples))
    num_samples_train = int(num_samples*ratio_train_test)
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train:]
    X_train,Y_train = X[idx_train], y[idx_train]
    X_test,Y_test = X[idx_test], y[idx_test]
    # --------------------------------------------------- """

    # ----------------------MNIST------------------------
    X_train, Y_train, X_test, Y_test = load()
    # ---------------------------------------------------


    num_trees = 10
    max_depth = 10 # maxim nombre nivells arbre
    min_size_split = 5 # si elements al node < 5 ja no dividim
    ratio_samples = 0.8 # bagging
    num_trees = 10
    criterion = "Gini"
    num_features_node = int(np.sqrt(X_train.shape[1])) # nombre de features diferents a consisderar
                                                       # en cada norain

    num_samples_train = X_train.shape[0]
    num_samples_test = X_test.shape[0]
    logger.info("{} train and {} test samples".format(num_samples_train, num_samples_test))

    try:
        start = timeit.default_timer()
        rf = RandomForest(max_depth,min_size_split,ratio_samples,num_trees,num_features_node,criterion)
        
        # ----------------------MNIST------------------------
        rf.values = range(0, 156, 64)
        # ---------------------------------------------------
        rf.fit(X_train, Y_train)
        #print("Fit is done")
        Ypred = rf.predict(X_test)
        stop = timeit.default_timer()
        execution_time = (stop - start) / 60.
        logger.info("Program Executed in " + str(execution_time) + " minutes.")
        num_correct_predictions = np.sum(Ypred == Y_test)
        accuracy = num_correct_predictions/float(len(Y_test))
        logger.info('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
        logger.info("Ypred = {}".format(Ypred))
        logger.info("Y_test = {}".format(Y_test))
        logger.info("Y_test - Y_train = {}".format(np.array([Y_test[i] - Ypred[i] for i in range(len(Y_test))])))

    except Exception as e:
        logger.critical(
            "Failed on executing due to:\n{}".format(str(e)))

if __name__ == "__main__":
    main()
