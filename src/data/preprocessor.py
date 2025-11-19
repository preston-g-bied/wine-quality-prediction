"""
Data preprocessor
"""
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from pathlib import Path
from src.data.data_loader import DataLoader
from config import *


class Preprocessor():
    def __init__(self, wine_type: str):
        self.wine_type = wine_type
        self.data_loader = DataLoader()

        self.scaler: Optional[FeatureScaler] = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'Preprocessor':
        self.scaler = FeatureScaler()
        self.scaler.fit(X_train)
        self.is_fitted = True

        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                  standardize: bool = True) -> Tuple[pd.Dataframe, Optional[pd.Series]]:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        X_transformed = X.copy()

        if standardize and self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)

        y_transformed = None
        if y is not None:
            y_transformed = y.map(QUALITY_TO_NUMERIC)

        return X_transformed, y_transformed
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series,
                      standardize: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train, standardize=standardize)
    
    def inverse_transform_labels(self, y_numeric: pd.Series) -> pd.Series:
        return y_numeric.map(QUALITY_LABELS)
    
    def save_preprocessor(self, filepath: Optional[Path] = None) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / f'{self.wine_type}_preprocessor.pkl'

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f'Saved preprocessor to {filepath}')

    @classmethod
    def load_preprocessor(cls, filepath: Path) -> 'Preprocessor':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
    def preprocess_and_save(self, standardize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = self.data_loader.load_splits(self.wine_type)

        self.fit(X_train, y_train)

        X_train_proc, y_train_proc = self.transform(X_train, y_train, standardize=standardize)
        X_test_proc, y_test_proc = self.transform(X_test, y_test, standardize=standardize)

        self._save_processed_data(X_train_proc, X_test_proc, y_train_proc, y_test_proc)

        self.save_preprocessor()

        return X_train_proc, X_test_proc, y_train_proc, y_test_proc
    
    def _save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series) -> None:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(PROCESSED_DATA_DIR / f'{self.wine_type}_X_train.csv', index=False)
        X_test.to_csv(PROCESSED_DATA_DIR / f'{self.wine_type}_X_test.csv', index=False)
        y_train.to_csv(PROCESSED_DATA_DIR / f'{self.wine_type}_y_train.csv', index=False, header=True)
        y_test.to_csv(PROCESSED_DATA_DIR / f'{self.wine_type}_y_test.csv', index=False, header=True)

class FeatureScaler():
    """StandardScaler wrapper"""

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