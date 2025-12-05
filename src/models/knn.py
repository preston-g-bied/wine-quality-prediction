"""
K-Nearest Neighbors model.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.models.base_model import BaseModel

class KNNModel(BaseModel):
    """
    Multi-class K-Nearest Neighbors Classifier wrapper
    """

    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski', p=2):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p  # euclidean
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        print("Feature importance not directly available for KNN")
        
        return pd.DataFrame({
            'feature': ['N/A'],
            'importance': [0.0]
        })
    
    def get_name(self) -> str:
        return f'KNN (k={self.n_neighbors}, weights={self.weights})'
    
    def get_params(self) -> dict:
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'p': self.p
        }