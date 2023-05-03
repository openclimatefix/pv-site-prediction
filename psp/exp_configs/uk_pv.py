"""Config used to train a model based on the `uk_pv` dataset."""

import datetime as dt
import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.dataset import Splits, split_train_test
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

# import multiprocessing
# import xgboost as xgb

PV_DATA_PATH = "data/5min.netcdf"
NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"

# A list of SS_ID that don't contain enough data.
# I just didn't want to calculate them everytime.
# TODO Get rid of those when we prepare the dataset.
SKIP_SS_IDS = [
    str(x)
    for x in [
        8440,
        16718,
        8715,
        17073,
        9108,
        9172,
        10167,
        10205,
        10207,
        10278,
        26778,
        26819,
        10437,
        10466,
        26915,
        10547,
        26939,
        26971,
        10685,
        10689,
        2638,
        2661,
        2754,
        2777,
        2783,
        2786,
        2793,
        2812,
        2829,
        2830,
        2867,
        2883,
        2904,
        2923,
        2947,
        2976,
        2989,
        2999,
        3003,
        3086,
        3118,
        3123,
        3125,
        3264,
        3266,
        3271,
        3313,
        3334,
        3470,
        3502,
        11769,
        11828,
        11962,
        3772,
        11983,
        3866,
        3869,
        4056,
        4067,
        4116,
        4117,
        4124,
        4323,
        4420,
        20857,
        4754,
        13387,
        13415,
        5755,
        5861,
        5990,
        6026,
        6038,
        6054,
        14455,
        6383,
        6430,
        6440,
        6478,
        6488,
        6541,
        6548,
        6560,
        14786,
        6630,
        6804,
        6849,
        6868,
        6870,
        6878,
        6901,
        6971,
        7055,
        7111,
        7124,
        7132,
        7143,
        7154,
        7155,
        7156,
        7158,
        7201,
        7237,
        7268,
        7289,
        7294,
        7311,
        7329,
        7339,
        7379,
        7392,
        7479,
        7638,
        7695,
        7772,
        15967,
        7890,
        16215,
        # This one has funny night values.
        7830,
    ]
]


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power"},
            ignore_pv_ids=SKIP_SS_IDS,
        )

    @functools.cache
    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_source=NwpDataSource(
                NWP_DATA_PATH,
                coord_system=27700,
                time_dim_name="init_time",
                value_name="UKV",
            )  # , cache_dir=".nwp_cache"),
            # nwp_data_source=None,
        )

    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            config=PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=48 * 4)),
            **self.get_data_source_kwargs(),
            regressor=SklearnRegressor(
                num_train_samples=4096,
                normalize_targets=True,
                #
                # We have done some tests with xgboost and keep this as an example but note that we
                # haven't added xgboost to our list of dependencies.
                #
                # sklearn_regressor=xgb.XGBRegressor(
                #     objective='reg:pseudohubererror',
                #     eta=0.1,
                #     n_estimators=200,
                #     max_depth=5,
                #     min_child_weight=20,
                #     tree_method='hist',
                #     n_jobs=multiprocessing.cpu_count() // 2,
                # ),
            ),
            use_nwp=True,
            # Those are the variables available in our prod environment.
            nwp_variables=[
                "si10",
                "vis",
                # "r2",
                "t",
                "prate",
                # "sd",
                "dlwrf",
                "dswrf",
                "hcc",
                "mcc",
                "lcc",
            ],
            normalize_features=True,
            use_inferred_meta=False,
        )

    def make_dataset_splits(self, pv_data_source: PvDataSource) -> Splits:
        return split_train_test(
            pv_data_source,
            # Starting in 2020 because we only have NWP data from 2020.
            # TODO Get the NWP data for 2018 and 2019.
            # train_start = datetime(2018, 1, 1)
            train_start=dt.datetime(2020, 1, 1),
            # Leaving a couple of days at the end to be safe.
            train_end=dt.datetime(2020, 12, 29),
            test_start=dt.datetime(2021, 1, 1),
            test_end=dt.datetime(2022, 1, 1),
        )
