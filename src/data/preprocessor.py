"""
Data preprocessor
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from src.data.data_loader import DataLoader
from config import *


class Preprocessor():
    def __init__(self, wine_type: str, standardize_data: bool = True, save_data: bool = True):
        self.standardize_data = standardize_data
        self.save_data = save_data

        self.wine_type = wine_type
        self.data_loader = DataLoader()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_splits(wine_type)

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.y_train = self.y_train.map(QUALITY_TO_NUMERIC)
        self.y_test = self.y_test.map(QUALITY_TO_NUMERIC)

        if self.standardize_data:
            self.standardize()

        if self.save_data:
            self.save_processed_data()

        return self.X_train, self.X_test, self.y_train, self.y_test

    def standardize(self) -> None:
        scaler = FeatureScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def save_processed_data(self) -> None:
        self.X_train.to_csv(f'{SPLITS_DIR}/{self.wine_type}_X_train_processed.csv', index=False)
        self.X_test.to_csv(f'{SPLITS_DIR}/{self.wine_type}_X_test_processed.csv', index=False)
        self.y_train.to_csv(f'{SPLITS_DIR}/{self.wine_type}_y_train_processed.csv', index=False)
        self.y_test.to_csv(f'{SPLITS_DIR}/{self.wine_type}_y_test_processed.csv', index=False)

class FeatureScaler():
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> None:
        self.feature_names = X_train.columns.tolist()
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)
    
    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform.")
        
        X_original = self.scaler.inverse_transform(X_scaled)
        return pd.DataFrame(X_original, columns=self.feature_names, index=X_scaled.index)
    
# could add ClassBalancer and PCA classes later