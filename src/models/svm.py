"""
Support Vector Machine Model
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from src.models.base_model import BaseModel

class SVMModel(BaseModel):
    """
    Multi-class Support Vector Machine wrapper
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=None):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=self.random_state,
            probability=True    # for predict_proba
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
        
        if self.kernel == 'linear':
            feature_names = self.model.feature_names_in_
            coefficients = np.abs(self.model.coef_).mean(axis=0)

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            })

            return importance_df.sort_values('importance', ascending=False)
        else:
            print(f"Feature importance not directly available for {self.kernel} kernel")
            print(f"Number of support vectors: {len(self.model.support_vectors_)}")
            return pd.DataFrame({
                'feature': ['N/A'],
                'importance': [0.0]
            })
        
    def get_name(self) -> str:
        return f'SVM (C={self.C}, kernel={self.kernel})'
    
    def get_params(self) -> dict:
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'random_state': self.random_state
        }