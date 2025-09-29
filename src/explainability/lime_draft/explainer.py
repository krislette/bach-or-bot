import numpy as np
from functools import partial
from sklearn.utils import check_random_state
import sklearn.metrics
from typing import List, Tuple, Any
import warnings
import librosa

from lime_draft.base import LimeBase


class IndexedString:
    # Minimal IndexedString that segments by LINE for lyrical explanations
    def __init__(
        self, raw_string: str, bow: bool = True, split_expression=None, mask_string=None
    ):
        self.raw = raw_string
        # split into lines by newline; keep non-empty lines
        self._words = [ln.strip() for ln in raw_string.split("\n") if ln.strip()]
        self.mask_string = mask_string if mask_string is not None else ""

    def num_words(self) -> int:
        return len(self._words)

    def word(self, i: int) -> str:
        return self._words[i]

    def inverse_removing(self, inactive_indexes: np.ndarray) -> str:
        if inactive_indexes is None or len(inactive_indexes) == 0:
            return "\n".join(self._words)
        inactive_set = set(inactive_indexes.tolist())
        kept = [w for idx, w in enumerate(self._words) if idx not in inactive_set]
        return "\n".join(kept)


class TextDomainMapper:
    def __init__(self, indexed_string: IndexedString):
        self.indexed_string = indexed_string


class MultimodalExplanation(object):
    def __init__(
        self,
        indexed_string,
        factorization,
        neighborhood_data,
        neighborhood_labels,
        modality,
    ):
        self.factorization = factorization
        self.indexed_string = indexed_string
        self.neighborhood_data = neighborhood_data
        self.neighborhood_labels = neighborhood_labels
        self.modality = modality
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}
        self.distance = {}

    def get_sorted_components(
        self,
        label,
        positive_components=True,
        negative_components=True,
        num_components="all",
        min_abs_weight=0.0,
        return_indeces=False,
    ):
        if label not in self.local_exp:
            raise KeyError("Label not in explanation")
        if positive_components is False and negative_components is False:
            raise ValueError(
                "positive_components, negative_components or both must be True"
            )
        n_audio_features = self.factorization.get_number_components()
        exp = self.local_exp[label]

        w = [[x[0], x[1]] for x in exp]
        used_features, weights = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1]

        if not negative_components:
            pos_weights = np.argwhere(weights > 0)[:, 0]
            used_features = used_features[pos_weights]
            weights = weights[pos_weights]
        elif not positive_components:
            neg_weights = np.argwhere(weights < 0)[:, 0]
            used_features = used_features[neg_weights]
            weights = weights[neg_weights]
        if min_abs_weight != 0.0:
            abs_weights = np.argwhere(abs(weights) >= min_abs_weight)[:, 0]
            used_features = used_features[abs_weights]
            weights = weights[abs_weights]

        if num_components == "all":
            num_components = len(used_features)
        else:
            assert isinstance(num_components, int)

        used_features = used_features[:num_components]
        weights = weights[:num_components]
        components = []
        for index in used_features:
            if self.modality == "both":
                if index < n_audio_features:
                    components.append(
                        self.factorization.get_ordered_component_names()[index]
                    )
                else:
                    components.append(
                        self.indexed_string.word(index - n_audio_features)
                    )
            elif self.modality == "lyrical":
                components.append(self.indexed_string.word(index))
            elif self.modality == "audio":
                components.append(
                    self.factorization.get_ordered_component_names()[index]
                )

        if return_indeces:
            return components, weights, used_features
        return components, weights


class LimeMusicExplainer(object):
    def __init__(
        self,
        kernel_width=25,
        kernel=None,
        verbose=False,
        class_names=None,
        feature_selection="auto",
        absolute_feature_sort=False,
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
        random_state=None,
        char_level=False,
    ):
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        # use our LimeBase
        self.base = LimeBase(kernel_fn, verbose, absolute_feature_sort)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    def explain_instance(
        self,
        factorization,
        text_instance,
        predict_fn,
        labels=None,
        top_labels=None,
        num_reg_targets=None,
        num_features=100000,
        num_samples=1000,
        batch_size=10,
        distance_metric="cosine",
        model_regressor=None,
        random_seed=None,
        fit_intercept=True,
        modality="both",
    ):
        is_classification = False
        if labels or top_labels:
            is_classification = True
        if is_classification and num_reg_targets:
            raise ValueError(
                "Set labels or top_labels for classification. "
                "Set num_reg_targets for regression."
            )
        if modality not in ["both", "lyrical", "audio"]:
            raise ValueError('Set modality arguement to "both", "lyrical" or "audio".')

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        self.factorization = factorization
        top = labels

        # use our line-based IndexedString
        indexed_string = IndexedString(
            text_instance, bow=self.bow, mask_string=self.mask_string
        )
        domain_mapper = TextDomainMapper(indexed_string)

        data, labels_out, distances = self.combined_data_labels_distances(
            indexed_string,
            predict_fn,
            num_samples,
            batch_size=batch_size,
            distance_metric=distance_metric,
            modality=modality,
        )

        ret_exp = MultimodalExplanation(
            indexed_string, self.factorization, data, labels_out, modality=modality
        )

        if top_labels:
            top = np.argsort(labels_out[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                data,
                labels_out,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )

        return ret_exp

    def combined_data_labels_distances(
        self,
        indexed_string,
        predict_fn,
        num_samples,
        modality,
        batch_size=10,
        distance_metric="cosine",
    ):

        doc_size = indexed_string.num_words()
        audio_size = self.factorization.get_number_components()

        if modality == "both":
            total_features = doc_size + audio_size
        elif modality == "lyrical":
            total_features = doc_size
        elif modality == "audio":
            total_features = audio_size

        data = self.random_state.randint(0, 2, num_samples * (total_features)).reshape(
            (num_samples, total_features)
        )
        data[0, :] = 1

        labels = []
        audios = []
        texts = []

        for row in data:
            if modality == "both":
                non_zeros = np.where(row[:audio_size] != 0)[0]
                temp = self.factorization.compose_model_input(non_zeros)
                audios.append(temp)

                inactive = np.where(row[audio_size:] == 0)[0]
                perturbed_string = indexed_string.inverse_removing(inactive)
                texts.append(perturbed_string)

            if modality == "audio":
                non_zeros = np.where(row != 0)[0]
                temp = self.factorization.compose_model_input(non_zeros)
                audios.append(temp)

                inactive = np.array([])
                perturbed_string = indexed_string.inverse_removing(inactive)
                texts.append(perturbed_string)

            if modality == "lyrical":
                all_oness = np.ones(audio_size)
                non_zeros = np.where(all_oness != 0)[0]
                temp = self.factorization.compose_model_input(non_zeros)
                audios.append(temp)

                inactive = np.where(row == 0)[0]
                perturbed_string = indexed_string.inverse_removing(inactive)
                texts.append(perturbed_string)

            if len(audios) == batch_size:
                preds = predict_fn(texts, np.array(audios))
                labels.extend(preds)
                audios = []
                texts = []

        if len(audios) > 0:
            preds = predict_fn(texts, np.array(audios))
            labels.extend(preds)

        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        return data, np.array(labels), distances
