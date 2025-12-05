"""
AdaBoost model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.models.base_model import BaseModel

class AdaBoostModel(BaseModel):
    """
    Multi-class AdaBoost Classifier wrapper
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, 
                 base_estimator_max_depth=1, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator_max_depth = base_estimator_max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Create base estimator (decision stump or shallow tree)
        base_estimator = DecisionTreeClassifier(
            max_depth=self.base_estimator_max_depth,
            random_state=self.random_state
        )
        
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            algorithm='SAMME'
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
        return f'AdaBoost (n_estimators={self.n_estimators}, lr={self.learning_rate})'
    
    def get_params(self) -> dict:
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'base_estimator_max_depth': self.base_estimator_max_depth,
            'random_state': self.random_state
        }