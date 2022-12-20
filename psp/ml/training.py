import os
from operator import itemgetter
from typing import TypedDict

import numpy as np
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe

from psp.data.data_sources.pv import PvDataSource
from psp.ml.dataset import PvXDataPipe, get_y_from_x
from psp.ml.models.base import FeaturesType, PvSiteModel
from psp.ml.types import FutureIntervals, PvId, Timestamp, X, Y


# This is the kind of object that goes through the pipeline!
class Sample(TypedDict):
    x: X
    y: Y
    features: FeaturesType


# Using classes for function factories to make pickle happy.
class _to_dict:
    def __init__(self, key):
        self.key = key

    def __call__(self, value):
        return {self.key: value}


class _add_y:
    def __init__(self, future_intervals, data_source):
        self.future_intervals = future_intervals
        self.data_source = data_source

    def __call__(self, sample: Sample):
        y = get_y_from_x(
            sample["x"],
            future_intervals=self.future_intervals,
            data_source=self.data_source,
        )
        return sample | {"y": y}


class _add_features:
    def __init__(self, model: PvSiteModel):
        self.model = model

    def __call__(self, sample):
        features = self.model.get_features(sample["x"])
        return sample | {"features": features}


def _y_is_not_none(sample: Sample):
    return sample["y"] is not None and not np.isnan(sample["y"].powers).all()


class _RandomSkip(IterDataPipe):
    """Randomly skip samples."""
    def __init__(
        self,
        datapipe: IterDataPipe,
        prob_keep: float,
        random_state: np.random.RandomState,
    ):
        self._datapipe = datapipe
        self._prob_keep = prob_keep
        self._random_state = random_state

    def __iter__(self):
        for x in self._datapipe:
            if self._random_state.random() < self._prob_keep:
                yield x


def make_data_loader(
    *,
    data_source: PvDataSource,
    pv_ids: list[PvId],
    future_intervals: FutureIntervals,
    min_ts: Timestamp,
    max_ts: Timestamp,
    prob_keep_sample: float = 1.0,
    random_state: np.random.RandomState | None,
    model: PvSiteModel,
    # Defaults to `os.cpu_count() // 2`
    num_workers: int | None = None
) -> DataLoader[Sample]:

    if num_workers is None:
        num_workers = (os.cpu_count() or 0) // 2

    datapipe = PvXDataPipe(
        data_source=data_source,
        future_intervals=future_intervals,
        pv_ids=pv_ids,
        min_ts=min_ts,
        max_ts=max_ts,
    )

    if prob_keep_sample < 1:
        assert random_state is not None
        datapipe = _RandomSkip(
            datapipe=datapipe,
            prob_keep=prob_keep_sample,
            random_state=random_state,
        )

    # This has to be as early as possible to be efficient!
    datapipe = datapipe.sharding_filter()

    # Put in a dictionary.
    datapipe = datapipe.map(_to_dict("x"))

    # Add 'y'.
    datapipe = datapipe.map(_add_y(future_intervals, data_source))
    datapipe = datapipe.filter(_y_is_not_none)

    # Add the features defined by the model.
    datapipe = datapipe.map(_add_features(model))

    data_loader: DataLoader[Sample] = DataLoader(
        datapipe,
        num_workers=num_workers,
        # TODO Actually support batches.
        collate_fn=itemgetter(0),
    )

    return data_loader
