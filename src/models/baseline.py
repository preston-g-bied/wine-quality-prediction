"""
Baseline models
"""

import pandas as pd
import numpy as np
import config
from src.models.base_model import BaseModel

class BaselineModel(BaseModel):
    # predicts the majority class only
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        modes = y_train.mode()
        self.majority_class = modes.iloc[0]
        self.is_fitted = True

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = [self.majority_class] * X_test.shape[0]
        return pd.Series(preds).reset_index(drop=True)
    
    def get_name(self) -> str:
        return "Baseline Model"
    
class StratifiedBaselineModel(BaseModel):
    # predicts classes with the same probability as class priors
    def __init__(self):
        self.probs = {}
        self.random_seed = config.RANDOM_SEED

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        class_priors = y_train.value_counts(normalize=True)
        for y_class in y_train.unique():
            self.probs[y_class] = class_priors[y_class]
        self.is_fitted = True

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        np.random.seed(self.random_seed)
        preds = []
        for i in range(X_test.shape[0]):
            random_pred = np.random.choice(list(self.probs.keys()), p=list(self.probs.values()))
            preds.append(random_pred)
        return pd.Series(preds).reset_index(drop=True)
    
    def get_name(self) -> str:
        return "Stratified Baseline Model"