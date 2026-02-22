import pandas as pd
import numpy as np
from src.utils import get_mappings, load_params

config = load_params()

def load_and_merge(train_path, store_path):
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, how='left', on='Store')
    return df

def clean_data(df):
    # Fix StateHoliday mixed types (0 vs '0')
    df['StateHoliday'] = df['StateHoliday'].astype(str)

    # Fill missing values based on logic
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna("None")

    # Drop rows where store is closed (Sales=0) to focus on active health
    df = df[df['Open'] != 0]
    return df.copy()

def feature_engineering(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # ensuring reproducibility
    mappings = get_mappings()
    for col in config['categorical_features']:
        df[col] = df[col].map(lambda x: mappings.get(x, 0))

    return df

def split_data(df):
    # Sort by Date and reset index
    df = df.sort_values('Date').reset_index(drop=True)

    # Features and Target
    X = df[config['training_data']['features']]
    y = df[config['training_data']['target']]

    # splitting ratio of 70:15:15
    total_rows = len(df)
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['cv_ratio']
    # Test ratio is the remainder 0.15

    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    # Train: Start to 70%
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    # Validation: 70% to 85%
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    # Test: 85% to End
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test