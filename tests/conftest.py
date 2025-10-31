import pytest
import pandas as pd


@pytest.fixture
def sample_df(tmp_path):
    df = pd.DataFrame({
        "user_idx": [0, 1, 0, 2],
        "item_idx": [10, 11, 12, 13],
        "label": [1, 0, 1, 0]
    })
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)
    return path
