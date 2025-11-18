"""
Trains baseline models
"""
import sys
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('..')
from src.data.data_loader import DataLoader
from src.models.baseline import BaselineModel, StratifiedBaselineModel

def evaluate_baseline(y_test, preds) -> None:
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print(f"Precision: {precision_score(y_test, preds, average='weighted', zero_division=0)}")
    print(f"Recall: {recall_score(y_test, preds, average='weighted', zero_division=0)}")
    print(f"F1: {f1_score(y_test, preds, average='weighted', zero_division=0)}")

def train_baseline(wine_type: str, model: Union[BaselineModel, StratifiedBaselineModel]) -> None:
    print(f'TRAINING {model.get_name().upper()} for {wine_type} wines')
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_splits(wine_type)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    evaluate_baseline(y_test, preds)

def main():
    wine_types = ['red', 'white']

    for wine_type in wine_types:
        baseline_model = BaselineModel()
        stratified_baseline_model = StratifiedBaselineModel()
        models = [baseline_model, stratified_baseline_model]
        for model in models:
            train_baseline(wine_type, model)
            print("\n")

if __name__ == "__main__":
    main()