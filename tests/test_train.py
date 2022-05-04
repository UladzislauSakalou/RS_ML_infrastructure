from typing import Any
from click.testing import CliRunner
import pytest

from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


path_test_data = [(True, 1), (4, 1), (2.5, 1)]


@pytest.mark.parametrize("path, expected_output", path_test_data)
def test_error_for_dataset_path(
    runner: CliRunner, path: Any, expected_output: int
) -> None:
    """It fails when dataset path is wrong"""
    result = runner.invoke(
        train,
        [
            "--dataset-path",
            path,
        ],
    )
    assert result.exit_code == expected_output


@pytest.mark.parametrize("path, expected_output", path_test_data)
def test_error_for_save_model_path(
    runner: CliRunner, path: Any, expected_output: int
) -> None:
    """It fails when save model path is wrong"""
    result = runner.invoke(
        train,
        [
            "--save-model-path",
            path,
        ],
    )
    assert result.exit_code == expected_output


model_name_test_data = [(True, 2), (4, 2), (2.5, 2), ("knn", 2)]


@pytest.mark.parametrize("model_name, expected_output", model_name_test_data)
def test_error_for_model_name(
    runner: CliRunner, model_name: Any, expected_output: int
) -> None:
    """It fails when model name is wrong"""
    result = runner.invoke(
        train,
        [
            "--model-name",
            model_name,
        ],
    )
    assert result.exit_code == expected_output
    assert "Invalid value for '--model-name'" in result.output


random_state_test_data = [
    ("test", [2, "'test' is not a valid integer."]),
    ("abc", [2, "'abc' is not a valid integer."]),
]


@pytest.mark.parametrize("random_state, expected_output", random_state_test_data)
def test_error_random_state(
    runner: CliRunner, random_state: Any, expected_output: Any
) -> None:
    """It fails when random state is wrong"""
    result = runner.invoke(
        train,
        [
            "--random-state",
            random_state,
        ],
    )
    assert result.exit_code == expected_output[0]
    assert expected_output[1] in result.output


use_scaler_test_data = [
    ("test", [2, "'test' is not a valid boolean."]),
    ("abc", [2, "'abc' is not a valid boolean."]),
    (4.5, [1, ""]),
    (4, [1, ""]),
]


@pytest.mark.parametrize("use_scaler, expected_output", use_scaler_test_data)
def test_error_use_scaler(
    runner: CliRunner, use_scaler: Any, expected_output: Any
) -> None:
    """It fails when random state is wrong"""
    result = runner.invoke(
        train,
        [
            "--use-scaler",
            use_scaler,
        ],
    )
    assert result.exit_code == expected_output[0]
    assert expected_output[1] in result.output


use_boruta_test_data = [
    ("test", [2, "'test' is not a valid boolean."]),
    ("abc", [2, "'abc' is not a valid boolean."]),
    (4.5, [1, ""]),
    (4, [1, ""]),
]


@pytest.mark.parametrize("use_boruta, expected_output", use_boruta_test_data)
def test_error_use_boruta(
    runner: CliRunner, use_boruta: Any, expected_output: Any
) -> None:
    """It fails when random state is wrong"""
    result = runner.invoke(
        train,
        [
            "--use-boruta",
            use_boruta,
        ],
    )
    assert result.exit_code == expected_output[0]
    assert expected_output[1] in result.output
