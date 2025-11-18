"""
Data loaders for red/white wine.
"""

import config
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.red_path = config.RED_WINE_PATH
        self.white_path = config.WHITE_WINE_PATH

    def load_wine_data(self, wine_type: str) -> pd.DataFrame:
        """
        Loads wine data by color
        wine_type is either 'red' or 'white'
        return dataframe for wine color
        """
        if wine_type == 'red':
            path = self.red_path
        elif wine_type == 'white':
            path = self.white_path
        else:
            raise ValueError(f"wine_type must be 'red' or 'white', got '{wine_type}'")
        
        df = pd.read_csv(path, sep=';')
        return df
    
    def add_quality_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a new column for binned quality
        """
        df['quality_class'] = pd.cut(
            df['quality'],
            bins=config.BIN_EDGES,
            labels=config.BIN_LABELS,
            right=False,
            include_lowest=True
        )
        return df
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Seperate X from y and split into train/test
        """
        X = df.drop(columns=['quality', 'quality_class'])
        y = df['quality_class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            stratify=y,
            random_state=config.RANDOM_SEED
        )
        return X_train, X_test, y_train, y_test
    
    def save_splits(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, wine_type: str) -> None:
        X_train.to_csv(path=f'{config.SPLITS_DIR}/{wine_type}_X_train.csv', index=False)
        X_test.to_csv(path=f'{config.SPLITS_DIR}/{wine_type}_X_test.csv', index=False)
        y_train.to_csv(path=f'{config.SPLITS_DIR}/{wine_type}_y_train.csv', index=False)
        y_test.to_csv(path=f'{config.SPLITS_DIR}/{wine_type}_y_test.csv', index=False)

    def load_splits(self, wine_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train = pd.read_csv(f'{config.SPLITS_DIR}/{wine_type}_X_train.csv')
        X_test = pd.read_csv(f'{config.SPLITS_DIR}/{wine_type}_X_test.csv')
        y_train = pd.read_csv(f'{config.SPLITS_DIR}/{wine_type}_y_train.csv').squeeze()
        y_test = pd.read_csv(f'{config.SPLITS_DIR}/{wine_type}_y_test.csv').squeeze()
        return X_train, X_test, y_train, y_test