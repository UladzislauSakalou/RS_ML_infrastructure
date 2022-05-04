from typing import Any
from click.testing import CliRunner
import pytest
from faker import Faker
from forest_ml.train import train
import pandas as pd
from joblib import load
import numpy as np
from forest_ml.pipeline import create_pipeline

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

def test_valid_scenario(runner: CliRunner):
    with runner.isolated_filesystem():
        fake_data = generate_fake_data(100)
        fake_data.to_csv('train.csv', index=False)
    
        result = runner.invoke(
            train,
            [
                "--dataset-path",
                'train.csv',
                "--save-model-path",
                'model.joblib'
            ]
        )
        
        model = load('model.joblib')
        assert hasattr(model, 'predict') == True
        assert hasattr(model, 'predict_proba') == True
        
        test_raws = generate_fake_data(50).drop(['Id', 'Cover_Type'], axis=1)
        pipeline = create_pipeline(use_scaler=True, use_boruta=False)
        test_raws = pipeline.fit_transform(test_raws)
        
        assert np.all(model.predict_proba(test_raws) >= 0) == True
        assert np.all(model.predict_proba(test_raws) <= 1) == True
        
        assert np.all(model.predict(test_raws)>= 1) == True
        assert np.all(model.predict(test_raws)<= 7) == True
    
        assert result.exit_code == 0

def generate_fake_data(num_rows: int):
    fake = Faker()
    Faker.seed(0)
    output = [{"Id":fake.unique.random_int(min=1, max=10000),
                "Elevation":fake.random_int(min=2376, max=3849),
                "Aspect":fake.random_int(min=0, max=360),
                "Slope":fake.random_int(min=0, max=52),
                "Horizontal_Distance_To_Hydrology":fake.random_int(min=0, max=1343),
                "Vertical_Distance_To_Hydrology":fake.random_int(min=-146, max=554),
                "Horizontal_Distance_To_Roadways":fake.random_int(min=0, max=6890),
                "Hillshade_9am":fake.random_int(min=0, max=254),
                "Hillshade_Noon":fake.random_int(min=99, max=254),
                "Hillshade_3pm":fake.random_int(min=0, max=248),
                "Horizontal_Distance_To_Fire_Points":fake.random_int(min=0, max=6993),
                "Wilderness_Area1":fake.random_int(min=0, max=1),
                "Wilderness_Area2":fake.random_int(min=0, max=1),
                "Wilderness_Area3":fake.random_int(min=0, max=1),
                "Wilderness_Area4":fake.random_int(min=0, max=1),
                "Soil_Type1":fake.random_int(min=0, max=1),
                "Soil_Type2":fake.random_int(min=0, max=1),
                "Soil_Type3":fake.random_int(min=0, max=1),
                "Soil_Type4":fake.random_int(min=0, max=1),
                "Soil_Type5":fake.random_int(min=0, max=1),
                "Soil_Type6":fake.random_int(min=0, max=1),
                "Soil_Type7":fake.random_int(min=0, max=1),
                "Soil_Type8":fake.random_int(min=0, max=1),
                "Soil_Type9":fake.random_int(min=0, max=1),
                "Soil_Type10":fake.random_int(min=0, max=1),
                "Soil_Type11":fake.random_int(min=0, max=1),
                "Soil_Type12":fake.random_int(min=0, max=1),
                "Soil_Type13":fake.random_int(min=0, max=1),
                "Soil_Type14":fake.random_int(min=0, max=1),
                "Soil_Type15":fake.random_int(min=0, max=1),
                "Soil_Type16":fake.random_int(min=0, max=1),
                "Soil_Type17":fake.random_int(min=0, max=1),
                "Soil_Type18":fake.random_int(min=0, max=1),
                "Soil_Type19":fake.random_int(min=0, max=1),
                "Soil_Type20":fake.random_int(min=0, max=1),
                "Soil_Type21":fake.random_int(min=0, max=1),
                "Soil_Type22":fake.random_int(min=0, max=1),
                "Soil_Type23":fake.random_int(min=0, max=1),
                "Soil_Type24":fake.random_int(min=0, max=1),
                "Soil_Type25":fake.random_int(min=0, max=1),
                "Soil_Type26":fake.random_int(min=0, max=1),
                "Soil_Type27":fake.random_int(min=0, max=1),
                "Soil_Type28":fake.random_int(min=0, max=1),
                "Soil_Type29":fake.random_int(min=0, max=1),
                "Soil_Type30":fake.random_int(min=0, max=1),
                "Soil_Type31":fake.random_int(min=0, max=1),
                "Soil_Type32":fake.random_int(min=0, max=1),
                "Soil_Type33":fake.random_int(min=0, max=1),
                "Soil_Type34":fake.random_int(min=0, max=1),
                "Soil_Type35":fake.random_int(min=0, max=1),
                "Soil_Type36":fake.random_int(min=0, max=1),
                "Soil_Type37":fake.random_int(min=0, max=1),
                "Soil_Type38":fake.random_int(min=0, max=1),
                "Soil_Type39":fake.random_int(min=0, max=1),
                "Soil_Type40":fake.random_int(min=0, max=1),
                "Cover_Type":fake.random_int(min=1, max=7)} for x in range(num_rows)]
    
    return pd.DataFrame(output, columns = get_columns())
    
def get_columns():
    return [
        "Id",
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",
        "Soil_Type1",
        "Soil_Type2",
        "Soil_Type3",
        "Soil_Type4",
        "Soil_Type5",
        "Soil_Type6",
        "Soil_Type7",
        "Soil_Type8",
        "Soil_Type9",
        "Soil_Type10",
        "Soil_Type11",
        "Soil_Type12",
        "Soil_Type13",
        "Soil_Type14",
        "Soil_Type15",
        "Soil_Type16",
        "Soil_Type17",
        "Soil_Type18",
        "Soil_Type19",
        "Soil_Type20",
        "Soil_Type21",
        "Soil_Type22",
        "Soil_Type23",
        "Soil_Type24",
        "Soil_Type25",
        "Soil_Type26",
        "Soil_Type27",
        "Soil_Type28",
        "Soil_Type29",
        "Soil_Type30",
        "Soil_Type31",
        "Soil_Type32",
        "Soil_Type33",
        "Soil_Type34",
        "Soil_Type35",
        "Soil_Type36",
        "Soil_Type37",
        "Soil_Type38",
        "Soil_Type39",
        "Soil_Type40",
        "Cover_Type"
    ]
    
    
    
									
