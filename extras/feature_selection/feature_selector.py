from feature_selection.correlation_metric import PearsonCorrelation, MutualInformation, SignalNoiseRatio, Silhouette, \
    ChiSquare


class FeatureSelector(object):
    def __init__(self, selection_method, k):
        self.selection_method = selection_method
        self.k = k

        if (selection_method == "pearson"):
            self.correlation_metric = PearsonCorrelation()
        elif (selection_method == "mi"):
            self.correlation_metric = MutualInformation()
        elif (selection_method == "snr"):
            self.correlation_metric = SignalNoiseRatio()
        elif (selection_method == "silhouette"):
            self.correlation_metric = Silhouette()
        elif (selection_method == "chi2"):
            self.correlation_metric = ChiSquare()
        else:
            self.correlation_metric = None

    def fit(self, dataset, datalabels):
        if not self.correlation_metric:
            print("{} method not known".format(self.selection_method))
            return

        self.correlations_ = sorted(self.correlation_metric.get_correlation(dataset, datalabels),
                                    key=lambda row: -row[1])[:self.k]

        return self

    def transform(self, dataset):
        return [[xi[corr[0]] for corr in self.correlations_] for xi in dataset]


class SelectionCombinator(object):
    def __init__(self, selector_combinations):
        self.feature_selectors = {}
        for selection_method, dims in selector_combinations:
            self.feature_selectors[selection_method] = FeatureSelector(selection_method, dims)

    def get_reduced_data(self, train_data, validation_data, test_data, all_labels):
        i = 0
        for selection_method in self.feature_selectors:
            i += 1
            feature_selector = self.feature_selectors[selection_method].fit(train_data + validation_data, all_labels)
            train_data = feature_selector.transform(train_data)
            validation_data = feature_selector.transform(validation_data)
            test_data = feature_selector.transform(test_data)

            if i == 1:
                self.selected_features_ = [corr[0] for corr in feature_selector.correlations_]
            else:
                feats = [self.selected_features_[corr[0]] for corr in feature_selector.correlations_]
                self.selected_features_ = feats.copy()

        return train_data, validation_data, test_data
