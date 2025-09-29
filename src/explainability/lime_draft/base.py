import numpy as np
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data."""

    def __init__(
        self, kernel_fn, verbose=False, absolute_feature_sort=False, random_state=None
    ):
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.absolute_feature_sort = absolute_feature_sort
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Run LARS path algorithm for feature selection with lasso."""
        x_vector = weighted_data
        alphas, _, coefs = lars_path(
            x_vector, weighted_labels, method="lasso", verbose=False
        )
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model."""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -1e9
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(
                    data[:, used_features + [feature]], labels, sample_weight=weights
                )
                score = clf.score(
                    data[:, used_features + [feature]], labels, sample_weight=weights
                )
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. See explain_instance_with_data."""
        if method == "none":
            return np.array(range(data.shape[1]))

        elif method == "forward_selection":
            return self.forward_selection(data, labels, weights, num_features)

        elif method == "highest_weights":
            clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            weighted_data = coef * data[0]
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]),
                reverse=True,
            )
            return np.array([x[0] for x in feature_weights[:num_features]])

        elif method == "lasso_path":
            weighted_data = (
                data - np.average(data, axis=0, weights=weights)
            ) * np.sqrt(weights[:, np.newaxis])
            weighted_labels = (labels - np.average(labels, weights=weights)) * np.sqrt(
                weights
            )
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features

        elif method == "auto":
            if num_features <= 6:
                n_method = "forward_selection"
            else:
                n_method = "highest_weights"
            return self.feature_selection(data, labels, weights, num_features, n_method)

    def explain_instance_with_data(
        self,
        neighborhood_data,
        neighborhood_labels,
        distances,
        label,
        num_features,
        feature_selection="auto",
        model_regressor=None,
        fit_intercept=True,
    ):
        """Generates explanations for a given instance and its perturbed neighborhood."""
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        used_features = self.feature_selection(
            neighborhood_data, labels_column, weights, num_features, feature_selection
        )

        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1, fit_intercept=fit_intercept, random_state=self.random_state
            )

        easy_model = model_regressor
        easy_model.fit(
            neighborhood_data[:, used_features], labels_column, sample_weight=weights
        )

        prediction_score = easy_model.score(
            neighborhood_data[:, used_features], labels_column, sample_weight=weights
        )

        local_pred = easy_model.predict(
            neighborhood_data[0, used_features].reshape(1, -1)
        )

        if self.absolute_feature_sort:
            sorted_local_exp = sorted(
                zip(used_features, easy_model.coef_),
                key=lambda x: np.abs(x[1]),
                reverse=True,
            )
        else:
            sorted_local_exp = sorted(
                zip(used_features, easy_model.coef_), key=lambda x: x[1], reverse=True
            )

        if self.verbose:
            print("Intercept:", easy_model.intercept_)
            print("Prediction_local:", local_pred)
            print("Right:", neighborhood_labels[0, label])
            print("Score:", prediction_score)

        return (easy_model.intercept_, sorted_local_exp, prediction_score, local_pred)
