"""Config used to train a model based on the `uk_pv` dataset."""

import datetime as dt

import numpy as np

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import NetcdfPvDataSource, PvDataSource
from psp.data_sources.irradiance import ZarrIrradianceDataSource, IrradianceDataSource
from psp.dataset import PvSplits, auto_date_split, split_pvs
from psp.exp_configs.base import ExpConfigBase
from psp.models.base import PvSiteModel, PvSiteModelConfig
from psp.models.recent_history import RecentHistoryModel
from psp.models.regressors.decision_trees import SklearnRegressor
from psp.typings import Horizons

# import multiprocessing
# import xgboost as xgb

PV_DATA_PATH = "/run/media/jacob/data/5min_v3.nc"
#PV_DATA_PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/clients/uk_pv/5min_v3.nc"
# NWP_DATA_PATH = "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_7.zarr"
NWP_DATA_PATHS = [
    (
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP"
        f"/UK_Met_Office/UKV/zarr/UKV_{year}_NWP.zarr"
    )
    for year in range(2020, 2021)
]

IRRADIANCE_DATA_PATH = "/run/media/jacob/data/xarray_train/combined.zarr"
#IRRADIANCE_DATA_PATH = "/home/jacob/combined.zarr"
# TODO Change this to all others but these
SS_ID_TO_KEEP = [10426, 10512, 10528, 10548, 10630, 10639, 10837, 11591, 12642, 12846,
                 12847, 12860, 14577, 14674, 16364, 16474, 17166, 26771, 26772, 26786,
                 26795, 26818, 26835, 26837, 26866, 26870, 26898, 26911, 26928, 26935,
                 26944, 26951, 26954, 26955, 26963, 26965, 26978, 26983, 26991, 26994,
                 27003, 27012, 27014, 27046, 27047, 27053, 27056, 2881, 2915, 3005,
                 3026, 3248, 3250, 3263, 3324, 3333, 3437, 3489, 3805, 3951,
                 4065, 5512, 5900, 6011, 6330, 6410, 6433, 6442, 6481, 6493,
                 6504, 6620, 6633, 6641, 6645, 6656, 6676, 6807, 6880, 6991,
                 6998, 7076, 7177, 7194, 7234, 7238, 7247, 7255, 7256, 7349,
                 7390, 7393, 7464, 7487, 7527, 7537, 7557, 7595, 7720, 7762,
                 7845, 7906, 7932, 8137, 8591, 8856, 8914, 9101, 9107, 9191,
                 9760]

# A list of SS_ID that don't contain enough data.
# I just didn't want to calculate them everytime.
# TODO Get rid of those when we prepare the dataset.
# TODO Change this back after irradiance testing, this is all non-irradiance IDs now
SKIP_SS_IDS = [
    str(x)
    for x in [4114, 4116, 4117, 8215, 12314, 4124, 4127, 8229, 8253, 4157, 4158, 4159, 8266, 8267, 12368, 12371, 8281, 16477, 16480, 16483, 12398, 4238, 12451, 8376, 16570, 12487, 12495, 16597, 8411, 8412, 8419, 8420, 4323, 4326, 8440, 26917, 8453, 8464, 4392, 8505, 4420, 4421, 4422, 16715, 16717, 16718, 12644, 12645, 12646, 8551, 12647, 12648, 8558, 20857, 4476, 4480, 16769, 16770, 8587, 8612, 8648, 12761, 12764, 8670, 12766, 12772, 8708, 8713, 8715, 16920, 16921, 12826, 12886, 12887, 8801, 12917, 12918, 12919, 12920, 17034, 17035, 4754, 17062, 17073, 13012, 13052, 13057, 27037, 27038, 9069, 9070, 9108, 27048, 17333, 9153, 9171, 9172, 9173, 13308, 13309, 13310, 13311, 17465, 5177, 13387, 13388, 13389, 13390, 13415, 9350, 5265, 9366, 9367, 9368, 9369, 13485, 13498, 13499, 13500, 5352, 5358, 5372, 13570, 9478, 9479, 9480, 13607, 5427, 5428, 5429, 9530, 9531, 5444, 13641, 5471, 9569, 9570, 13670, 9648, 13767, 13768, 13769, 13770, 13773, 5589, 17878, 5596, 5597, 5598, 13811, 13817, 9730, 13840, 5660, 9763, 9764, 9765, 5690, 9815, 9816, 18019, 5753, 5754, 5755, 9865, 9866, 9867, 9870, 9871, 5778, 5780, 5781, 5803, 5805, 9902, 9903, 5826, 18128, 5861, 9960, 5871, 18161, 5890, 9989, 5895, 5899, 10003, 10004, 10005, 5908, 6875, 22335, 10048, 10049, 5953, 18249, 10063, 10064, 5974, 5976, 10082, 5986, 10086, 5990, 6021, 6026, 6032, 10131, 6035, 6038, 6049, 10149, 10150, 6054, 6064, 10167, 10168, 10169, 6075, 6093, 10190, 6094, 10205, 10206, 10207, 14308, 14316, 6125, 10222, 6126, 6127, 10254, 10278, 10280, 14394, 14452, 14453, 14455, 10361, 10366, 10367, 14467, 26773, 26774, 26775, 26776, 26777, 26778, 26780, 26781, 26782, 26783, 26784, 26785, 26787, 26788, 26789, 26790, 26791, 26792, 26793, 26794, 26796, 26797, 26798, 26799, 26800, 26801, 26802, 26803, 26804, 26805, 26806, 26807, 26808, 10425, 26809, 26811, 26812, 26813, 6329, 26815, 26816, 26817, 6336, 26819, 14531, 10437, 10438, 26822, 10440, 26824, 26826, 26827, 26823, 26829, 26830, 26831, 26825, 26833, 26828, 26832, 26834, 26836, 26838, 26839, 26840, 26841, 26842, 26843, 26844, 26845, 26846, 6362, 26848, 26849, 10466, 26851, 26852, 26853, 26850, 26855, 26856, 26854, 26858, 26859, 26860, 26861, 26862, 26863, 26864, 6380, 6383, 26867, 26868, 26869, 26871, 26872, 26873, 26874, 26875, 26876, 26877, 26878, 26879, 26880, 26881, 26882, 10497, 26883, 26885, 26884, 26887, 26886, 26889, 26890, 26891, 26892, 10509, 26893, 26894, 26895, 10513, 26896, 26899, 26900, 26901, 26897, 26902, 26903, 26904, 26905, 26906, 10523, 26908, 26907, 26909, 26910, 26912, 26914, 26913, 26915, 10531, 10532, 10533, 26916, 26918, 26922, 26923, 26924, 26919, 26920, 26921, 26925, 26926, 26927, 26929, 26932, 10547, 26934, 26933, 26936, 26930, 26931, 26937, 26938, 26939, 26940, 26941, 26942, 26943, 26945, 26947, 26948, 26949, 14657, 26952, 26953, 14666, 6472, 26956, 26957, 26958, 26959, 14672, 14673, 26960, 26961, 26962, 26964, 26966, 26967, 26968, 10585, 26970, 10586, 26969, 26971, 10589, 26972, 26973, 26974, 26975, 26979, 26980, 26981, 26976, 10595, 26977, 26982, 26984, 26985, 26986, 26987, 26988, 26989, 26990, 26992, 26993, 26995, 6516, 26996, 26997, 26998, 26999, 27000, 27001, 27002, 10619, 10620, 6526, 6527, 27004, 27005, 27006, 27007, 27008, 27009, 27013, 10631, 27016, 27017, 27018, 27011, 27015, 6541, 27020, 27021, 27022, 27025, 27026, 10640, 6548, 27023, 27024, 27027, 27028, 27029, 27030, 27035, 27031, 10648, 10649, 10650, 6560, 27032, 27033, 27034, 27036, 14757, 6566, 6567, 27041, 27042, 27043, 6571, 27044, 6573, 27045, 6575, 6576, 6577, 27049, 27050, 27051, 27052, 27054, 27055, 27057, 27058, 27059, 27060, 27061, 27062, 27063, 27064, 18873, 27065, 27066, 27067, 10685, 10686, 10689, 14786, 10692, 10693, 6596, 6597, 6601, 6602, 6603, 6605, 10702, 10704, 6609, 6610, 6613, 6614, 6615, 6616, 6617, 6618, 6619, 6621, 6629, 6630, 6634, 6637, 6638, 6640, 6643, 6646, 6648, 6663, 6665, 6667, 14859, 6669, 6670, 6671, 6672, 6673, 14861, 6675, 6677, 6678, 6681, 6682, 10791, 10792, 10793, 10794, 2603, 23083, 18989, 18990, 2607, 2625, 2626, 6721, 2628, 6723, 6726, 2631, 6727, 6729, 14923, 6732, 14924, 2638, 10835, 10838, 10840, 10841, 10842, 10843, 10844, 6748, 2657, 2660, 2661, 6780, 6781, 6785, 6786, 6789, 10890, 6794, 6795, 6796, 6797, 6798, 6800, 6801, 6804, 6815, 6817, 2729, 6826, 6827, 6828, 6830, 10929, 6834, 6835, 6838, 6839, 6842, 6843, 6845, 6846, 6848, 6849, 2754, 2760, 6856, 15051, 2766, 6862, 6865, 2770, 6867, 6868, 6869, 6870, 2775, 2776, 2777, 6874, 6871, 6872, 6873, 10973, 10974, 2783, 2784, 10975, 2786, 10976, 2789, 6877, 6878, 6881, 2793, 6882, 6891, 6892, 6896, 6897, 6898, 15091, 6901, 6902, 2812, 2813, 2814, 2815, 2816, 2819, 6917, 2822, 2824, 2828, 2829, 2830, 2832, 2833, 2834, 2835, 6928, 6932, 6936, 6938, 11040, 11042, 6949, 6950, 2855, 6951, 6952, 6953, 6958, 6961, 6962, 2867, 6964, 6966, 6967, 6971, 6972, 6975, 2880, 2883, 6979, 6981, 6982, 6990, 6994, 2902, 2903, 2904, 6999, 7001, 2908, 2912, 2918, 7014, 7016, 7017, 2923, 7019, 7025, 7030, 7033, 15225, 7035, 2940, 2947, 7043, 7044, 7049, 7050, 7051, 11151, 7055, 11153, 7058, 7060, 7061, 7062, 7065, 7066, 11166, 2975, 2976, 11174, 11175, 11176, 2989, 7085, 7088, 7090, 7092, 2997, 2998, 2999, 3000, 3001, 7093, 3003, 3007, 7111, 3016, 3017, 7114, 3019, 3020, 3021, 7116, 7117, 7118, 7119, 7120, 7124, 7127, 7132, 3039, 7136, 7143, 7149, 7151, 7152, 7154, 7155, 7156, 7158, 7159, 7162, 7163, 7166, 15359, 3074, 7171, 7172, 7173, 7174, 7175, 3080, 7176, 3085, 3086, 3087, 7184, 3089, 3093, 3094, 11287, 7190, 7192, 7195, 3100, 7199, 7200, 7201, 7202, 7203, 3117, 3118, 7213, 3122, 3123, 3125, 3126, 7229, 7230, 7236, 7237, 7239, 7240, 7241, 3146, 3147, 3148, 3149, 7243, 7245, 3152, 7248, 7257, 7258, 7259, 7262, 7268, 3175, 7274, 7275, 7276, 7277, 7285, 7286, 7287, 7289, 7290, 7291, 7292, 7293, 7294, 7295, 3208, 11401, 7311, 7312, 7313, 7314, 7315, 7317, 3231, 3233, 7329, 3235, 3236, 3237, 3238, 3239, 3240, 3242, 3243, 7338, 7339, 11438, 7344, 3249, 11441, 7351, 3258, 7356, 7359, 3264, 3266, 3267, 3268, 3270, 3271, 7368, 11465, 11466, 7369, 3278, 3280, 3281, 7378, 7379, 7383, 3288, 7384, 7386, 7387, 7392, 7394, 7395, 7396, 11494, 7399, 7401, 7402, 3311, 7408, 3313, 7409, 7410, 7411, 7412, 15603, 3323, 3325, 3326, 3329, 11522, 3334, 3338, 6363, 7438, 7439, 7440, 6364, 7445, 7446, 7447, 7448, 7451, 7452, 11558, 7468, 7469, 7471, 7473, 11573, 7477, 7478, 7479, 6372, 7490, 11590, 7495, 7498, 7517, 7519, 7520, 7521, 3431, 3432, 3433, 7528, 3435, 11628, 15721, 11630, 7533, 7534, 3442, 3443, 3444, 7547, 7548, 3453, 3454, 7550, 7551, 7553, 7554, 7555, 7556, 7558, 11655, 11656, 11657, 7560, 7562, 3469, 3470, 7566, 3472, 3473, 7568, 7569, 3476, 15765, 7579, 7580, 7581, 3487, 3488, 7585, 11685, 11687, 3496, 11688, 7593, 3502, 7603, 7608, 3513, 11712, 11724, 3536, 7633, 7634, 7635, 3540, 7636, 7638, 3543, 7639, 7640, 15830, 7642, 7644, 7648, 7651, 15846, 11752, 15850, 15851, 7660, 7661, 7662, 15852, 7664, 7669, 6409, 7670, 11769, 7674, 7693, 7694, 7695, 7699, 7700, 7702, 11811, 7719, 7721, 7722, 7723, 11827, 11828, 6424, 7748, 6425, 7756, 7757, 7758, 7759, 6427, 15949, 7763, 7766, 7767, 11865, 7772, 6430, 15967, 7776, 7777, 6431, 11895, 11896, 6440, 10511, 7830, 6441, 7832, 7833, 7834, 7835, 7836, 7837, 7843, 7844, 7846, 7847, 6446, 7865, 3770, 11962, 3772, 7885, 6452, 11983, 7890, 11987, 3799, 7901, 7902, 7903, 7904, 7905, 3811, 7907, 7909, 7910, 26810, 12013, 3822, 3831, 7927, 7928, 7930, 12029, 12035, 3857, 3864, 3865, 3866, 3869, 3872, 12091, 12092, 6478, 16214, 16215, 16216, 6482, 3952, 8066, 6488, 6489, 16267, 6490, 6491, 6492, 8090, 6494, 4002, 4003, 8099, 16298, 6498, 4029, 4030, 4033, 4035, 4044, 4045, 4046, 6503, 4056, 6506, 4067, 14649, 14650, 8175, 4090, 4092]
]


def _get_capacity(d):
    # Use 0.99 quantile over the history window, fallback on the capacity as defined
    # in the metadata.
    value = float(d["power"].quantile(0.99))
    if not np.isfinite(value):
        value = float(d.coords["capacity"].values)
    return value


class ExpConfig(ExpConfigBase):
    def get_pv_data_source(self):
        return NetcdfPvDataSource(
            PV_DATA_PATH,
            id_dim_name="ss_id",
            timestamp_dim_name="timestamp",
            rename={"generation_wh": "power", "kwp": "capacity"},
            ignore_pv_ids=SKIP_SS_IDS,
        )

    def get_data_source_kwargs(self):
        return dict(
            pv_data_source=self.get_pv_data_source(),
            nwp_data_source=None,
            #NwpDataSource(
            #    NWP_DATA_PATHS,
            #    coord_system=27700,
            #    time_dim_name="init_time",
            #    value_name="UKV",
            #    y_is_ascending=False,
                # cache_dir=".nwp_cache",
            #),
            irradiance_data_source=ZarrIrradianceDataSource(
                IRRADIANCE_DATA_PATH,
            )
        )

    def _get_model_config(self) -> PvSiteModelConfig:
        return PvSiteModelConfig(horizons=Horizons(duration=15, num_horizons=48))

    def get_model(self, *, random_state: np.random.RandomState | None = None) -> PvSiteModel:
        kwargs = self.get_data_source_kwargs()
        return RecentHistoryModel(
            config=self._get_model_config(),
            **kwargs,
            regressor=SklearnRegressor(
                num_train_samples=4096*60,
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
            random_state=random_state,
            use_nwp=False,
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
            capacity_getter=_get_capacity,
            pv_dropout=0.1,
        )

    def make_pv_splits(self, pv_data_source: PvDataSource) -> PvSplits:
        return split_pvs(pv_data_source)

    def get_date_splits(self):
        return auto_date_split(
            test_start_date=dt.datetime(2021, 1, 1),
            test_end_date=dt.datetime(2021, 11, 8),
            train_days=356 * 3,
        )
