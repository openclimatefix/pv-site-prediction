import datetime as dt

from psp.models.base import PvSiteModel
from psp.typings import Features, X, Y


class MultiPvSiteModel(PvSiteModel):
    """Higher-order model that wrap a series of models trained at different intervals.

    When evaluating a given sample, we use the latest model that was trained before that sample.

    This wrapper class should not be used in production: use the child model directly instead.
    """

    def __init__(self, models: dict[dt.datetime, PvSiteModel]):
        self._models = models
        # Make sure the models are sorted by date.
        assert list(sorted(models)) == list(models)

    def predict_from_features(self, x: X, features: Features) -> Y:
        model = self._get_model_for_ts(x.ts)
        return model.predict_from_features(x, features)

    def get_features(self, x: X, is_training: bool = False) -> Features:
        model = self._get_model_for_ts(x.ts)
        return model.get_features(x, is_training=is_training)

    def _get_model_for_ts(self, ts: dt.datetime) -> PvSiteModel:
        # Use the most recent model whose train date is *before* `x.ts`. This was the most recent
        # model at time `x.ts`.
        for date, model in reversed(self._models.items()):
            if ts > date:
                return model
        else:
            raise ValueError(f"Date {ts} is before all the models")

    def set_data_sources(self, *args, **kwargs):
        for model in self._models.values():
            model.set_data_sources(*args, **kwargs)

    def explain(self, x: X):
        model = self._get_model_for_ts(x.ts)
        return model.explain(x)

    def get_train_date(self, ts: dt.datetime) -> dt.datetime:
        for date in reversed(self._models):
            if ts > date:
                return date
        else:
            raise ValueError(f"Date {ts} is before all the models")

    @property
    def config(self):
        # Assume all the configs are the same and return the first one.
        return next(iter(self._models.values())).config
