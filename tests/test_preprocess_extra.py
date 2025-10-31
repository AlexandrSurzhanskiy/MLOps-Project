import pandas as pd
import pytest
from src.data.preprocess import check_data_format


def test_check_data_format_raises_on_missing_columns():
    df = pd.DataFrame({"user_idx": [1, 2, 3], "item_idx": [4, 5, 6]})
    with pytest.raises(ValueError):
        check_data_format(df, required_cols=["user_idx", "item_idx", "label"])


def test_check_data_format_raises_on_wrong_type():
    df = pd.DataFrame(
        {"user_idx": ["a", "b"], "item_idx": [1, 2], "label": [1, 0]}
    )
    with pytest.raises((TypeError, ValueError)):
        check_data_format(df)
