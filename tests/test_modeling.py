import pandas as pd
from src.data_processing import preprocess_data


def test_preprocessing():
    df = pd.read_csv('../data/raw/train.csv')
    df = preprocess_data(df)
    assert df.isnull().sum().sum() == 0  # Проверка на отсутствие пропущенных значений

