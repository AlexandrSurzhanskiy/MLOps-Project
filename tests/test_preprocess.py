import pandas as pd
from src.data.preprocess import check_data_format


def test_check_data_format_valid(tmp_path):
    df = pd.DataFrame({"user_idx": [1,2], "item_idx": [3,4], "label": [0,1]})
    check_data_format(df)


def test_check_data_format_invalid(tmp_path):
    df = pd.DataFrame({"user_id": [1], "item_id": [2]})
    try:
        check_data_format(df)
    except ValueError as e:
        assert "user_idx" in str(e)
