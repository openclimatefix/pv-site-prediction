import logging
from itertools import islice
from typing import Any, Iterator, Tuple, overload

import numpy as np
import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor

from psp.models.regressors.base import Regressor
from psp.typings import Batch, BatchedFeatures, Features
from psp.utils.batches import batch_features, concat_batches

_log = logging.getLogger(__name__)


class ForestRegressor(Regressor):
    """Regressor that uses the HistGradientBoostingRegressor internally.

    We train only one regressor, that predicts all the outputs.
    """

    def __init__(self, num_train_samples: int):
        """
        Arguments:
        ---------
            num_samples: Number of samples to train the forest on.
        """
        # Using absolute error alleviates some outlier problems.
        # Squared loss (the default) makes the big outlier losses more important.
        self._tree = HistGradientBoostingRegressor(
            loss="absolute_error",
            random_state=1234,
        )
        self._num_train_samples = num_train_samples

    @overload
    def _prepare_features(self, features: BatchedFeatures) -> np.ndarray:
        ...

    @overload
    def _prepare_features(
        self, features: BatchedFeatures, feature_names: dict[str, list[str]]
    ) -> Tuple[np.ndarray, list[str]]:
        ...

    def _prepare_features(
        self,
        features: BatchedFeatures,
        feature_names: dict[str, list[str]] | None = None,
    ) -> np.ndarray | Tuple[np.ndarray, list[str]]:
        """Build a (sample, feature)-shaped matrix from (batched) features.

        Optionally also build a list of feature names that match the columns.

        Return:
        ------
            A numpy array (rows=sample, columns=features) and the name of the columns as a list of
            string.
        """
        per_horizon = features["per_horizon"]
        common = features["common"]

        # Start with `per_horizon`, to which we will add all the other features.
        new_features = per_horizon

        n_batch, n_horizon, n_features = new_features.shape
        (n_batch2, n_common_features) = common.shape
        assert n_batch == n_batch2

        if feature_names:
            col_names = feature_names["per_horizon"]

        # Add the horizon index as a feature.
        horizon_idx = np.broadcast_to(
            np.arange(n_horizon, dtype=float), (n_batch, n_horizon)
        ).reshape(n_batch, n_horizon, 1)

        # (batch * horizon, features + 1)
        new_features = np.concatenate([new_features, horizon_idx], axis=2)

        if feature_names:
            col_names.append("horizon_idx")

        common = np.broadcast_to(
            common.reshape(n_batch, 1, common.shape[1]),
            (n_batch, n_horizon, common.shape[1]),
        )

        new_features = np.concatenate([new_features, common], axis=2)

        if feature_names:
            col_names.extend(feature_names["common"])

        assert new_features.shape == (
            n_batch,
            n_horizon,
            n_features + 1 + n_common_features,
        )
        if feature_names:
            assert len(col_names) == n_features + 1 + n_common_features

        # Finally we flatten the horizons.
        new_features = new_features.reshape(n_batch * n_horizon, n_features + 1 + n_common_features)

        if feature_names:
            return new_features, col_names
        else:
            return new_features

    def train(
        self,
        train_iter: Iterator[Batch],
        valid_iter: Iterator[Batch],
        batch_size: int,
    ):
        num_samples = self._num_train_samples

        num_batches = num_samples // batch_size
        # We put `tqdm` here because that's the slow part that we can put a progress bar on.
        _log.info("Extracting the features.")
        batches = [b for b in tqdm.tqdm(islice(train_iter, num_batches), total=num_batches)]

        # Concatenate all the batches into one big batch.
        batch = concat_batches(batches)

        # Make it into a (sample, features)-shaped matrix.
        xs = self._prepare_features(batch.features)

        # (batch, horizon)
        poa = batch.features["poa_global"]
        assert len(poa.shape) == 2

        # (batch, 1)
        factor = np.array(batch.features["factor"]).reshape(-1, 1)
        assert factor.shape[0] == poa.shape[0]

        # (batch, horizon)
        ys = batch.y.powers

        # We can ignore the division by zeros, we treat the nan/inf later.
        with np.errstate(divide="ignore"):
            # No safe div because we want to ignore the points where `poa == 0`, we will multiply by
            # zero anyway later, so no need to learn that. Otherwise the model will spend a lot of
            # its capacity trying to learn that nights mean zero. Even if in theory it sounds like
            # a trivial decision to make for the model (if we use "poa_global" as a feature), in
            # practice ignoring those points seems to help a lot.
            ys = ys / (poa * factor)

        # Flatten the targets just like we "flattened" the features.
        ys = ys.reshape(-1)

        # Remove `nan`, `inf`, etc. from ys.
        mask = np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]

        _log.info("Fitting the forest.")
        self._tree.fit(xs, ys)

    def predict(self, features: Features):
        new_features = self._prepare_features(batch_features([features]))
        pred = self._tree.predict(new_features)
        return pred

    def explain(
        self, features: Features, feature_names: dict[str, list[str]]
    ) -> Tuple[Any, list[str]]:
        """Return the `shap` values for our sample, alonside the names of the features.

        We return a `shap` object that contains as many values as we have horizons for the sample
        (since internally we split those in individual samples before sending to the model).
        """
        try:
            import shap
        except ImportError:
            print("You need to install `shap` to use the `explain` functionality")
            return None, []

        batch = batch_features([features])

        new_features, new_feature_names = self._prepare_features(batch, feature_names)

        explainer = shap.Explainer(self._tree)
        shap_values = explainer(new_features)
        # TODO we should return something that is not shap-specific
        return shap_values, new_feature_names
