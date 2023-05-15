from click.testing import CliRunner

from psp.scripts.train_model import main


def _test_command(main_func, cmd_args: list[str]):
    runner = CliRunner()

    result = runner.invoke(main_func, cmd_args, catch_exceptions=True)

    # Without this the output to stdout/stderr is grabbed by click's test runner.
    print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception:
        raise result.exception

    assert result.exit_code == 0

    return result


def test_train_model(tmp_path):
    cmd_args = [
        "--exp-config-name",
        "test_config1",
        "--exp-root",
        str(tmp_path),
        "--exp-name",
        "train_test",
        "--batch-size",
        "1",
        "--num-test-samples",
        "10",
    ]

    _test_command(main, cmd_args)

    # Make sure a model was created.
    assert (tmp_path / "train_test" / "model_0.pkl").exists()
