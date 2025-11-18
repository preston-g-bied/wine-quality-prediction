"""
Trains baseline models
"""
import sys
from typing import Dict, Any
from src.evaluation.metrics import evaluate_model, print_evaluation

sys.path.append('..')
from src.data.data_loader import DataLoader
from src.models.base_model import BaseModel
from src.models.baseline import BaselineModel, StratifiedBaselineModel
from src.utils.experiment_tracker import ExperimentTracker

def train_baseline(wine_type: str, model: BaseModel) -> Dict[str, Any]:
    print(f'Training {model.get_name().upper()} for {wine_type} wines')
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_splits(wine_type)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = evaluate_model(y_test, preds, wine_type, model.get_name())
    print_evaluation(results)

    tracker = ExperimentTracker(model.get_name(), wine_type)
    tracker.log_experiment(results)

    return results

def main():
    wine_types = ['red', 'white']

    for wine_type in wine_types:
        baseline_model = BaselineModel()
        stratified_baseline_model = StratifiedBaselineModel()
        models = [baseline_model, stratified_baseline_model]
        for model in models:
            results = train_baseline(wine_type, model)

if __name__ == "__main__":
    main()