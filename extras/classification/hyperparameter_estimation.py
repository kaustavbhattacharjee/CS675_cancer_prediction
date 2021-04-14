from time import time
import pickle
from random import random

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import PredefinedSplit


class HyperparameterEstimation(object):
    def __init__(self, method, test_fold=None):
        self.method = method

        if test_fold:
            self.split = PredefinedSplit(test_fold)

    def _report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            import numpy as np
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def fit(self, params, X, y, pickle_file_path):

        if self.method == "svm":
            self.model = SVC()
        elif self.method == "naive-bayes":
            self.model = GaussianNB()
        elif self.method == "tree":
            self.model = DecisionTreeClassifier()
        else:
            self.model = None

        pickle_file_path = pickle_file_path[:-4] + "_cv_results.pkl"
        search_algo = params.pop('search')
        if search_algo == "random":
            n_iter_search = params.pop('n_iter')
            cv = self.split if hasattr(self, "split") else 5
            random_search = RandomizedSearchCV(self.model, param_distributions=params, n_iter=n_iter_search,
                                               cv=cv, n_jobs=-1, scoring="accuracy")

            st = time()
            random_search.fit(X, y)
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - st), n_iter_search))
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(random_search.cv_results_, f)

            self._report(random_search.cv_results_)

            self.model = random_search.best_estimator_
