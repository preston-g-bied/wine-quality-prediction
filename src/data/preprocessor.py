"""
Data preprocessor
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor():
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame) -> None:
        self.scaler.fit(X_train)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(X)