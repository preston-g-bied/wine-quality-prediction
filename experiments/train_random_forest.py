"""
Train and evaluate Random Forest model.
"""

import sys
sys.path.append('..')

from src.data.preprocessor import Preprocessor
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import (
    evaluate_model, print_evaluation,
    print_confusion_matrix, print_per_class_metrics
)
from src.utils.experiment_tracker import ExperimentTracker
from config import *

def train_random_forest(wine_type: str, n_estimators: int = 100, 
                       max_depth: int = None, min_samples_split: int = 2,
                       min_samples_leaf: int = 1):
    print(f'Training Random Forest for {wine_type.upper()} wines')
    print(f'Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, '
          f'min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}')

    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    model = RandomForestModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion='gini',
        random_state=RANDOM_SEED
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = evaluate_model(y_test, predictions, wine_type, model.get_name())
    print_evaluation(results)

    print_confusion_matrix(y_test, predictions)

    print_per_class_metrics(y_test, predictions)

    tracker = ExperimentTracker(model.get_name(), wine_type)
    tracker.log_experiment({
        **results,
        **model.get_params()
    })

    print("\nTop 5 Most Important Features:")
    feature_importance = model.get_feature_importance()
    print(feature_importance.head())

    return model, results

def main():
    wine_types = ['red', 'white']
    
    configs = [
        {'n_estimators': 50, 'max_depth': None},
        {'n_estimators': 100, 'max_depth': None},
        {'n_estimators': 200, 'max_depth': None},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 20},
    ]

    for wine_type in wine_types:
        for config in configs:
            model, results = train_random_forest(
                wine_type,
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=2,
                min_samples_leaf=1
            )

    print(f"\nResults saved to: {RESULTS_DIR / 'metrics'}")

if __name__ == "__main__":
    main()