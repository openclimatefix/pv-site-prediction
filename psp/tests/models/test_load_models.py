import pathlib

import pytest
from numpy.testing import assert_allclose

from psp.models.recent_history import SetupConfig
from psp.serialization import load_model
from psp.typings import X

# We ran the test once to get the results and pasted here the results.
# This way we can make sure the output doens't change.
EXPECTED_OUTPUT = {
    "model1": [1.0220418, 1.03555466, 1.04443434, 1.04860107, 1.04799241],
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
    model.setup(
        SetupConfig(
            pv_data_source=pv_data_source,
            nwp_data_source=None,
        )
    )

    pv_id = pv_data_source.list_pv_ids()[0]
    max_ = pv_data_source.max_ts()
    min_ = pv_data_source.min_ts()

    timestamp = min_ + (max_ - min_) / 2

    output = model.predict(X(pv_id=pv_id, ts=timestamp))

    assert_allclose(output.powers, expected, rtol=1e-8)
