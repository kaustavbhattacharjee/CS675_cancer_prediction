from time import time
from random import random, shuffle
import sys, os

from sklearn.svm import SVC

from lib.cli_helper import sanity_check
from lib.data_wrangler import DataWrangler
from lib.utility import Evaluation, combination_gen
from feature_selection.feature_selector import SelectionCombinator

print_prefix = "Time taken to complete"


def get_data():
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
        if random() > 0.3:
            training_set.append(xi)
            training_labels.append(label)
        else:
            validation_set.append(xi)
            validation_labels.append(label)

    combination = combination_gen(combination, len(train_data[0]), k)
    print(combination)

    return training_set, validation_set, test_data, list(list(zip(*training_labels))[0]), \
           list(list(zip(*validation_labels))[0]), combination


if __name__ == "__main__":


    super_start = time()
    train_data, validation_data, test_data, train_labels, validation_labels, combination = get_data()
    print("{} dataset load = {} seconds".format(print_prefix, time() - super_start))

    start = time()
    combinator = SelectionCombinator(combination)
    reduced_train_data, reduced_validation_data, reduced_test_data = combinator.get_reduced_data(train_data,
                                                                                                 validation_data,
                                                                                                 test_data,
                                                                                                 train_labels +
                                                                                                 validation_labels)
    del train_data
    del validation_data
    del test_data
    print("{} dataset reduction = {} seconds".format(print_prefix, time() - start))
    print("Selected features were {}".format(combinator.selected_features_))

    start = time()
    model = SVC(kernel="linear", C=3.0, max_iter=100000)
    predictions = model.fit(reduced_train_data, train_labels).predict(reduced_validation_data)
    accuracy = Evaluation.get_accuracy(predictions, validation_labels)
    print("{} model learning and predicting = {} seconds".format(print_prefix, time() - start))
    print("Got {} accuracy using linear SVM for 15 dims".format(accuracy))

    start = time()
    train_data = reduced_train_data + reduced_validation_data
    del reduced_train_data
    del reduced_validation_data
    train_labels = train_labels + validation_labels
    model = SVC(kernel="linear", C=3.0, max_iter=100000)
    predictions = model.fit(train_data, train_labels).predict(reduced_test_data)

    prediction_output_file = os.path.join(os.path.dirname(__file__), 'final_predictions.txt')
    text_to_write = "Selected features were:-\n{}\n".format(combinator.selected_features_)
    text_to_write =  text_to_write + "\n".join(["{} {}".format(predictions[i], i) for i in range(len(predictions))])
    DataWrangler.write_to_file(prediction_output_file, text_to_write)
    print("{} test prediction writing = {} seconds".format(print_prefix, time() - start))

    print("{} full processing = {} seconds".format(print_prefix, time() - super_start))
