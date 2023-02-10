import logging
from itertools import islice
from typing import Any, Iterator, Tuple, overload

import numpy as np
import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor

from psp.ml.models.regressors.base import Regressor
from psp.ml.typings import Batch, BatchedFeatures, Features
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
    ):
        """
        Return:
        -------
            A numpy array (rows=sample, columns=features) and the name of the columns as a list of
            string.
        """
        per_future = features["per_future"]
        common = features["common"]

        batch, fut, feat = per_future.shape

        # future * (batch, features)
        per_future_ = np.split(features["per_future"], per_future.shape[1], axis=1)

        if feature_names:
            col_names = feature_names["per_future"]

        per_future_ = [x.squeeze(1) for x in per_future_]

        # future * (batch, features + 1)
        per_future_ = [
            np.concatenate([x, np.broadcast_to(i, (x.shape[0], 1))], axis=1)
            for i, x in enumerate(per_future_)
        ]

        if feature_names:
            col_names.append("horizon_idx")

        # future * (batch, features + 1 + 2)
        per_future_ = [np.concatenate([x, common], axis=1) for x in per_future_]

        if feature_names:
            col_names.extend(["recent_power", "recent_power_is_nan"])

        # (future * batch, features + 1)
        new_features = np.concatenate(per_future_, axis=0)
        assert new_features.shape == (fut * batch, feat + 1 + 2)

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
        batches = [
            b for b in tqdm.tqdm(islice(train_iter, num_batches), total=num_batches)
        ]

        # Concatenate all the batches into one big batch.
        batch = concat_batches(batches)

        # Make it into a (sample, features)-shaped matrix.
        xs = self._prepare_features(batch.features)

        # (batch, future)
        poa = batch.features["poa_global"]
        assert len(poa.shape) == 2

        # (batch, 1)
        factor = np.array(batch.features["factor"]).reshape(-1, 1)
        assert factor.shape[0] == poa.shape[0]

        # (batch, future)
        ys = batch.y.powers

        # We can ignore the division by zeros, we treat the nan/inf later.
        with np.errstate(divide="ignore"):
            # No safe div because we want to ignore the points where `poa == 0`, we will multiply by
            # zero anyway later, so no need to learn that. Otherwise the model will spend a lot of
            # its capacity trying to learn that nights mean zero. Even if in theory it sounds like
            # a trivial decision to make for the model (if we use "poa_global" as a feature), in
            # practice ignoring those points seems to help a lot.
            ys = ys / (poa * factor)

        # future * (batch)
        ys_ = np.split(ys, ys.shape[1], axis=1)
        ys_ = [x.squeeze(1) for x in ys_]
        # (batch * future)
        ys = np.concatenate(ys_, axis=0)

        # Remove `nan`, `inf`, etc. from ys.
        mask = np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]

        _log.info("Fitting the forest.")
        self._tree.fit(xs, ys)

    def predict(self, features: Features):
        batched_features = batch_features([features])
        new_features = self._prepare_features(batched_features)
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
