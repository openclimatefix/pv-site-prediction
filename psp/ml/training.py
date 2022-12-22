import functools
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterDataPipe

from psp.data.data_sources.pv import PvDataSource
from psp.ml.dataset import DatasetSplit, PvXDataPipe, RandomPvXDataPipe, get_y_from_x
from psp.ml.typings import (
    Batch,
    BatchedFeatures,
    BatchedX,
    BatchedY,
    Features,
    FutureIntervals,
    Sample,
    X,
)


def _is_not_none(x):
    return x is not None


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


class _Limit(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, limit: int):
        self._dp = datapipe
        self._limit = limit

    def __iter__(self):
        for i, x in enumerate(self._dp):
            if i >= self._limit:
                return
            yield x


def _batch(samples: list[Sample]) -> Batch:
    assert len(samples) > 0
    x = BatchedX(
        pv_id=[s.x.pv_id for s in samples],
        ts=[s.x.ts for s in samples],
    )
    y = BatchedY(powers=np.stack([s.y.powers for s in samples]))
    features: BatchedFeatures = {
        key: np.stack([sample.features[key] for sample in samples])
        for key in samples[0].features
    }
    return Batch(x=x, y=y, features=features)


def _build_sample(
    x: X,
    *,
    future_intervals: FutureIntervals,
    data_source: PvDataSource,
    get_features: Callable[[X], Features],
) -> Sample | None:
    y = get_y_from_x(x, future_intervals=future_intervals, data_source=data_source)

    # Skip the heavy computation if the target doesn't make sense.
    if y is None:
        return None

    features = get_features(x)

    return Sample(x=x, y=y, features=features)


def make_data_loader(
    *,
    data_source: PvDataSource,
    future_intervals: FutureIntervals,
    split: DatasetSplit,
    get_features: Callable[[X], Features],
    prob_keep_sample: float = 1.0,
    random_state: np.random.RandomState | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    shuffle: bool = False,
    step: int = 1,
    limit: int | None = None,
) -> DataLoader[Sample]:
    """
    Arguments:
    --------
        step: Step in minutes for the timestamps.
        limit: return only this number of samples.
    """
    if prob_keep_sample < 1 and random_state is None:
        raise ValueError("You must provide a random state when sampling")

    pvx_datapipe: PvXDataPipe
    if shuffle:
        assert random_state is not None
        pvx_datapipe = RandomPvXDataPipe(
            data_source=data_source,
            future_intervals=future_intervals,
            random_state=random_state,
            pv_ids=split.pv_ids,
            start_ts=split.start_ts,
            end_ts=split.end_ts,
            step=step,
        )
    else:
        pvx_datapipe = PvXDataPipe(
            data_source=data_source,
            future_intervals=future_intervals,
            pv_ids=split.pv_ids,
            start_ts=split.start_ts,
            end_ts=split.end_ts,
            step=step,
        )

    if prob_keep_sample < 1:
        assert random_state is not None
        pvx_datapipe = _RandomSkip(
            datapipe=pvx_datapipe,
            prob_keep=prob_keep_sample,
            random_state=random_state,
        )

    # This has to be as early as possible to be efficient!
    pvx_datapipe = pvx_datapipe.sharding_filter()

    # This is the expensive part, where we compute our model-specific feature extraction.
    datapipe = pvx_datapipe.map(
        functools.partial(
            _build_sample,
            future_intervals=future_intervals,
            data_source=data_source,
            get_features=get_features,
        )
    )

    # `_build_sample` will return `None` when the sample is not useful (for instance when all the
    # targets have no data).
    datapipe = datapipe.filter(_is_not_none)

    # We add the ability to stop the pipeline after a `limit` number of samples.
    if limit is not None:
        datapipe = _Limit(datapipe, limit)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size, wrapper_class=_batch)

    data_loader: DataLoader2[Sample] = DataLoader2(
        datapipe, reading_service=MultiProcessingReadingService(num_workers=num_workers)
    )

    return data_loader
