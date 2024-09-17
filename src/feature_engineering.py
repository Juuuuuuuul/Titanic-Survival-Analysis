def create_features(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df
