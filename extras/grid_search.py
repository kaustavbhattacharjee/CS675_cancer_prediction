from time import time
from random import random, shuffle
import sys, os
import itertools
import pickle

from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint

from lib.cli_helper import sanity_check
from lib.data_wrangler import DataWrangler
from lib.utility import Evaluation, combination_gen
from feature_selection.feature_selector import SelectionCombinator
from feature_selection.utils import get_intersection
from classification.hyperparameter_estimation import HyperparameterEstimation

pickle_dir = os.path.join(os.path.dirname(__file__), 'pickle')
print_prefix = "Time taken to complete"


def set_up():
    k = 15
    combination = ('chi2', 'mi', 'snr', 'pearson', 'silhouette')

    train_data_file = os.path.join(pickle_dir, 'train_data.pkl')
    validation_data_file = os.path.join(pickle_dir, 'validation_data.pkl')
    test_data_file = os.path.join(pickle_dir, 'test_data.pkl')
    train_labels_file = os.path.join(pickle_dir, 'train_labels.pkl')
    validation_labels_file = os.path.join(pickle_dir, 'validation_labels.pkl')
    model_file = os.path.join(pickle_dir, 'hyperparameter_estimator.pkl')

    if os.path.exists(model_file):
        with open(train_data_file, 'rb') as f:
            training_set = pickle.load(f)
        with open(validation_data_file, 'rb') as f:
            validation_set = pickle.load(f)
        with open(test_data_file, 'rb') as f:
            test_data = pickle.load(f)
        with open(train_labels_file, 'rb') as f:
            training_labels = pickle.load(f)
        with open(validation_labels_file, 'rb') as f:
            validation_labels = pickle.load(f)
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        k, combination = sanity_check()

        train_data = DataWrangler.read_from_file(sys.argv[1], conversion=float)
        test_data = DataWrangler.read_from_file(sys.argv[2], conversion=float)
        train_labels = DataWrangler.read_from_file(sys.argv[3], conversion=int)
        zipped_train = list(zip(train_data, train_labels))
        shuffle(zipped_train)
        train_data, train_labels = zip(*zipped_train)
        training_set = []
        validation_set = []
        training_labels = []
        validation_labels = []
        for xi, label in zip(train_data, train_labels):
            if random() > 0.1:
                training_set.append(xi)
                training_labels.append(label)
            else:
                validation_set.append(xi)
                validation_labels.append(label)
        validation_fold = [1 if random() > 0.1 else 0 for xi in training_set]
        model = HyperparameterEstimation("svm", validation_fold)

        with open(train_data_file, 'wb') as f:
            pickle.dump(training_set, f)
        with open(validation_data_file, 'wb') as f:
            pickle.dump(validation_set, f)
        with open(test_data_file, 'wb') as f:
            pickle.dump(test_data, f)
        with open(train_labels_file, 'wb') as f:
            pickle.dump(training_labels, f)
        with open(validation_labels_file, 'wb') as f:
            pickle.dump(validation_labels, f)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

    return training_set, validation_set, test_data, list(list(zip(*training_labels))[0]), \
           list(list(zip(*validation_labels))[0]), model, k, combination


if __name__ == '__main__':
    super_start = time()
    best_model = None
    best_combination = None
    best_test_data = None
    best_accuracy = None
    train_data, validation_data, test_data, train_labels, validation_labels, model, k, combination = set_up()
    print("{} dataset load = {} seconds".format(print_prefix, time() - super_start))

    for i in range(len(combination)):
        for combination in itertools.combinations(metrics, i + 1):
            combination_start = time()
            combinations = combination_gen(combination, len(train_data[0]), k)

            red_feats_start = time()
            # Get dimension reduced dataset
            reduced_dataset_file_prefix = "{}_{}dims_".format(os.path.join(pickle_dir, '_'.join(combination)), k)
            if os.path.exists(reduced_dataset_file_prefix + "validation.pkl"):
                with open(reduced_dataset_file_prefix + "train.pkl", 'rb') as f:
                    reduced_train_data = pickle.load(f)
                with open(reduced_dataset_file_prefix + "test.pkl", 'rb') as f:
                    reduced_test_data = pickle.load(f)
                with open(reduced_dataset_file_prefix + "validation.pkl", 'rb') as f:
                    reduced_validation_data = pickle.load(f)
            else:
                combinator = SelectionCombinator(combinations)
                reduced_train_data, reduced_validation_data, reduced_test_data = \
                    combinator.get_reduced_data(train_data, validation_data, test_data,
                                                train_labels + validation_labels)
                with open(reduced_dataset_file_prefix + "train.pkl", 'wb') as f:
                    pickle.dump(reduced_train_data, f)
                with open(reduced_dataset_file_prefix + "test.pkl", 'wb') as f:
                    pickle.dump(reduced_test_data, f)
                with open(reduced_dataset_file_prefix + "validation.pkl", 'wb') as f:
                    pickle.dump(reduced_validation_data, f)

            print("{} dataset reduction to {} dimensions = {} seconds".format(print_prefix, k,
                                                                              time() - red_feats_start))

            start = time()
            # Run Hyperparameter optimization to get best params combination
            model_pickle_file_path = "{}_{}dims_svm.pkl".format('_'.join(combination), k)
            model_pickle_file_path = os.path.join(pickle_dir, model_pickle_file_path)
            if os.path.exists(model_pickle_file_path):
                with open(model_pickle_file_path, 'rb') as f:
                    model.model = pickle.load(f)
            else:
                param_dist = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                              "C": sp_randint(1, 10),
                              "degree": sp_randint(2, 6),
                              "search": "random",
                              "n_iter": 40}
                try:
                    model.fit(param_dist, reduced_train_data, train_labels, model_pickle_file_path)
                except Exception as e:
                    print("Error while running for {} file.".format(reduced_dataset_file_prefix))
                    print(len(reduced_train_data), len(train_labels))
                    print(e)
                    sys.exit(0)
                with open(model_pickle_file_path, 'wb') as f:
                    pickle.dump(model.model, f)
            predictions = model.model.predict(reduced_validation_data)
            accuracy = Evaluation.get_accuracy(predictions, validation_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = '_'.join(combination)
                best_model = model.model
                best_test_data = reduced_test_data.copy()
            print("{} hyperparameter estimation = {} seconds".format(print_prefix, time() - start))
            print("{} processing for {} dimensions = {} seconds".format(print_prefix, k, time() - red_feats_start))

            print("{} processing of combination set {} = {} seconds".format(print_prefix, '-'.join(combination),
                                                                            time() - combination_start))

    print("\n\n\n")
    start = time()
    # Write predictions to file
    prediction_output_file = os.path.join(os.path.dirname(__file__), 'final_predictions_{}dims.txt'.format(k))
    print("*" * 100)
    print("Got {} accuracy for {} combination on SVM classifier.".format(best_accuracy, best_combination))
    print("*" * 100)
    predictions = best_model.predict(best_test_data)
    DataWrangler.write_to_file(prediction_output_file, predictions)
    print("{} prediction writing = {} seconds".format(print_prefix, time() - start))
    print("\n\n\n")

    print("{} full processing = {} seconds".format(print_prefix, time() - super_start))
