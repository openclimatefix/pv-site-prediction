import pathlib

import pytest
from numpy.testing import assert_allclose

from psp.serialization import load_model
from psp.typings import X

# We ran the test once to get the results and pasted here the results.
# This way we can make sure the output doens't change.
EXPECTED_OUTPUT = {
    "model_v1": [1.0220418, 1.03555466, 1.04443434, 1.04860107, 1.04799241],
    "model_v2": [0.447506, 0.448072, 0.446772, 0.443611, 0.438599],
    "model_v3": [0.801439, 0.802452, 0.800124, 0.794462, 0.785487],
    "model_v4": [0.498424, 0.499054, 0.497606, 0.494085, 0.488503],
    "model_v5": [0.484041, 0.484653, 0.483247, 0.479827, 0.474406],
    "model_v6": [0.484041, 0.484653, 0.483247, 0.479827, 0.474406],
}


def test_old_models_sanity_check():
    """We simply make sure that we are testing all the models from the fixture directory."""
    models = [
        p.stem for p in pathlib.Path("psp/tests/fixtures/models").iterdir() if p.suffix == ".pkl"
    ]

    assert sorted(models) == sorted(EXPECTED_OUTPUT)


@pytest.mark.parametrize("model_path,expected", list(EXPECTED_OUTPUT.items()))
def test_old_models(model_path, expected, pv_data_source):
    """Make sure that we can load previously trained models."""
    model = load_model(f"psp/tests/fixtures/models/{model_path}.pkl")
    model.set_data_sources(
        pv_data_source=pv_data_source,
        nwp_data_source=None,
    )

    pv_id = pv_data_source.list_pv_ids()[0]
    max_ = pv_data_source.max_ts()
    min_ = pv_data_source.min_ts()

    timestamp = min_ + (max_ - min_) / 2

    output = model.predict(X(pv_id=pv_id, ts=timestamp))

    assert_allclose(output.powers, expected, atol=1e-6)
