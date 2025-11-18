"""
Abstract base class for all models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import pickle
from pathlib import Path

class BaseModel(ABC):
    """Base class that all models inherit from"""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return model name"""
        pass

    def save_model(self, filepath: Path) -> None:
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath: Path):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)