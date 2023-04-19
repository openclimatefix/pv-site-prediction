import functools

from psp.data.data_sources.nwp import NwpDataSource
from psp.data.data_sources.pv import NetcdfPvDataSource
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel, SetupConfig
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

# import multiprocessing
# import xgboost as xgb

PV_DATA_PATH = "data/5min.netcdf"
NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"

# TODO define
ExpConfigBase = object


class ExpConfig(ExpConfigBase):
    @functools.cache
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power"},
        )

    @functools.cache
    def get_model_setup_config(self):
        return SetupConfig(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_source=NwpDataSource(NWP_DATA_PATH)  # , cache_dir=".nwp_cache"),
            # nwp_data_source=None,
        )

    @functools.cache
    def _get_model_config(self):
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=48 * 4), blackout=0)

    @functools.cache
    def get_model(self) -> PvSiteModel:
        return RecentHistoryModel(
            self._get_model_config(),
            self.get_model_setup_config(),
            regressor=SklearnRegressor(
                num_train_samples=4096,
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
