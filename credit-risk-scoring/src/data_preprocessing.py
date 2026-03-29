import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path):
    df = pd.read_csv(path)
    return df
def preprocess_data(df):
    
    # drop useless column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # fill missing values
    df.fillna('unknown', inplace=True)

    # encode target
    if df['Risk'].dtype == 'object':
        df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})

    return df
def split_data(df):
    X = df.drop(columns=['Risk'])
    y = df['Risk']
    return train_test_split(X, y, test_size=0.2, random_state=42)
    
