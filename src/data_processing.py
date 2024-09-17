import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Cabin'].fillna('Unknown', inplace=True)  # Обработка пропусков в колонке 'Cabin'
    return df

