"""
Logistic Regression model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Multi-class Logistic Regression wrapper.
    """

    def __init__(self, C=1.0, max_iter=1000, random_state=None):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class='ovr',  # one vs. rest
            solver='lbfgs',
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
        coefficients = self.model.coef_

        importance_df = pd.DataFrame(
            coefficients.T,
            index=feature_names,
            columns=[f'Class_{i}' for i in range(coefficients.shape[0])]
        )

        importance_df['avg_importance'] = np.abs(coefficients).mean(axis=0)

        return importance_df.sort_values('avg_importance', ascending=False)
    
    def get_name(self) -> str:
        return f'Logistic Regression (C={self.C})'
    
    def get_params(self) -> dict:
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }