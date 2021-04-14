from math import sqrt


class Statistics:
    @classmethod
    def mean(cls, dataset):
        l = len(dataset)

        if isinstance(dataset[0], list):
            return [sum(dim) / l for dim in zip(*dataset)]

        return (sum(dataset) / l)

    @classmethod
    def variance(cls, dataset, mean_data=None):
        if mean_data is None:
            mean_data = cls.mean(dataset)

        l = len(dataset)
        if isinstance(dataset[0], list):
            return [sum([(i - m) ** 2 for i in xi]) / l for xi, m in zip(zip(*dataset), mean_data)]

        return sum((xi - mean_data) ** 2 for xi in dataset) / l

    @classmethod
    def covariance(cls, dataset_u, dataset_v, mean_u=None, mean_v=None):
        if mean_u is None:
            mean_u = cls.mean(dataset_u)
        if mean_v is None:
            mean_v = cls.mean(dataset_v)

        return sum((xi - mean_u) * (yi - mean_v) for xi, yi in zip(dataset_u, dataset_v)) / len(dataset_v)

    @classmethod
    def std_dev(cls, dataset, mean_data=None):
        if isinstance(dataset[0], list):
            return [sqrt(var) for var in cls.variance(dataset, mean_data)]

        return sqrt(cls.variance(dataset, mean_data))


class Evaluation:
    @classmethod
    def _01_accuracy(cls, predictions, true_labels):
        return sum(1.0 if yi == ri else 0.0 for yi, ri in zip(predictions, true_labels)) / len(predictions)

    @classmethod
    def get_accuracy(cls, predictions, true_labels, scoring="0-1"):
        if isinstance(scoring, str):
            if scoring == "0-1":
                scoring = cls._01_accuracy

        return scoring(predictions, true_labels)


class LinAlg:
    @classmethod
    def distance(cls, u, v):
        return sqrt(sum((ui - vi) ** 2 for ui, vi in zip(u, v)))

    @classmethod
    def dot_product(cls, vec_a, vec_b):
        condition = not (isinstance(vec_a, list) and isinstance(vec_b, list))
        if not condition:
            col_left = len(vec_a[0]) if isinstance(vec_a[0], list) else len(vec_a)
            row_right = len(vec_b)
            condition = col_left != row_right
        if condition:
            print("Dimensions of the vectors should be same for dot product")
            return None

        if not isinstance(vec_b[0], list):
            vec_b = list(zip(vec_b))
        if not isinstance(vec_a[0], list):
            vec_a = [vec_a]

        prod = []
        for ai in vec_a:
            pi = []
            for bi in zip(*vec_b):
                pij = 0.0
                for aij, bij in zip(ai, bi):
                    pij += aij * bij
                pi.append(pij)
            pi = pi[0] if len(pi) == 1 else pi
            prod.append(pi)

        return prod

    @classmethod
    def norm(cls, vec):
        return sqrt(cls.dot_product(vec, vec)[0])

    @classmethod
    def transpose(cls, X):
        if not (isinstance(X, list) and isinstance(X[0], list)):
            return None

        return [list(x) for x in zip(*X)]


def combination_gen(combination, n_features, k):
    l = len(combination)
    r = (n_features / k) ** (1 / l)

    return [(combination[i], int(n_features / (r ** (i + 1)))) for i in range(l - 1)] + [(combination[-1], k)]
