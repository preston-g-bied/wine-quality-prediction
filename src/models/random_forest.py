"""
Random Forest model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Multi-class Random Forest Classifier wrapper
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion='gini', random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
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
        
        feature_names = self.model.feature_names_in_
        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        return importance_df.sort_values('importance', ascending=False)
    
    def get_name(self) -> str:
        return f'Random Forest (n_estimators={self.n_estimators}, max_depth={self.max_depth})'
    
    def get_params(self) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'random_state': self.random_state
        }