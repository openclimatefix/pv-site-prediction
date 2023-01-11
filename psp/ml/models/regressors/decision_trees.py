import logging
from itertools import islice
from typing import Iterator

import numpy as np
import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor

from psp.ml.models.regressors.base import Regressor
from psp.ml.typings import Batch, BatchedFeatures
from psp.utils.maths import safe_div

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
        self._tree = HistGradientBoostingRegressor(loss="absolute_error")
        self._num_train_samples = num_train_samples

    def _prepare_features(self, per_future: np.ndarray, common: np.ndarray):
        batch, fut, feat = per_future.shape
        # future * (batch, features)
        per_future_ = np.split(per_future, per_future.shape[1], axis=1)
        per_future_ = [x.squeeze(1) for x in per_future_]
        # future * (batch, features + 1)
        per_future_ = [
            np.concatenate([x, np.broadcast_to(i, (x.shape[0], 1))], axis=1)
            for i, x in enumerate(per_future_)
        ]

        # future * (batch, features + 1 + 2)
        per_future_ = [np.concatenate([x, common], axis=1) for x in per_future_]

        # (future * batch, features + 1)
        per_future = np.concatenate(per_future_, axis=0)
        assert per_future.shape == (fut * batch, feat + 1 + 2)

        return per_future

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
            x for x in tqdm.tqdm(islice(train_iter, num_batches), total=num_batches)
        ]

        # (batch, future, features)
        per_future = np.concatenate([b.features["per_future"] for b in batches])
        common = np.concatenate([b.features["common"] for b in batches])
        per_future = self._prepare_features(per_future, common)

        # batch * (future,)
        irr_ = [b.features["irradiance"] for b in batches]
        # (batch, future)
        irr = np.vstack(irr_)

        # batch * ()
        factor_ = [b.features["factor"] for b in batches]
        # (batch, 1)
        factor = np.array(factor_).reshape(-1, 1)

        # (batch, future)
        ys = np.vstack([b.y.powers for b in batches])

        ys = safe_div(ys, irr * factor)

        # future * (batch)
        ys_ = np.split(ys, ys.shape[1], axis=1)
        ys_ = [x.squeeze(1) for x in ys_]
        # (batch * future)
        ys = np.concatenate(ys_, axis=0)

        # remove `nan` for ys
        mask = ~np.isnan(ys)
        per_future = per_future[mask]
        ys = ys[mask]

        _log.info("Fitting the forest.")
        self._tree.fit(per_future, ys)

    def predict(self, features: BatchedFeatures):
        per_future = features["per_future"]
        common = features["common"]
        per_future = np.expand_dims(per_future, axis=0)
        common = np.expand_dims(common, axis=0)
        features = self._prepare_features(per_future, common)
        pred = self._tree.predict(features)
        return pred
