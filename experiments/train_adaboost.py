"""
Train and evaluate AdaBoost model.
"""

import sys
sys.path.append('..')

from src.data.preprocessor import Preprocessor
from src.models.adaboost import AdaBoostModel
from src.evaluation.metrics import (
    evaluate_model, print_evaluation,
    print_confusion_matrix, print_per_class_metrics
)
from src.utils.experiment_tracker import ExperimentTracker
from config import *

def train_adaboost(wine_type: str, n_estimators: int = 50, 
                   learning_rate: float = 1.0, base_estimator_max_depth: int = 1):
    print(f'Training AdaBoost for {wine_type.upper()} wines')
    print(f'Hyperparameters: n_estimators={n_estimators}, '
          f'learning_rate={learning_rate}, '
          f'base_estimator_max_depth={base_estimator_max_depth}')

    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    model = AdaBoostModel(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        base_estimator_max_depth=base_estimator_max_depth,
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
        {'n_estimators': 50, 'learning_rate': 1.0, 'base_depth': 1},
        {'n_estimators': 100, 'learning_rate': 1.0, 'base_depth': 1},
        {'n_estimators': 200, 'learning_rate': 1.0, 'base_depth': 1},
        {'n_estimators': 100, 'learning_rate': 0.5, 'base_depth': 1},
        {'n_estimators': 100, 'learning_rate': 1.5, 'base_depth': 1},
        {'n_estimators': 100, 'learning_rate': 1.0, 'base_depth': 2},
        {'n_estimators': 100, 'learning_rate': 1.0, 'base_depth': 3},
    ]

    for wine_type in wine_types:
        for config in configs:
            model, results = train_adaboost(
                wine_type,
                n_estimators=config['n_estimators'],
                learning_rate=config['learning_rate'],
                base_estimator_max_depth=config['base_depth']
            )

    print(f"\nResults saved to: {RESULTS_DIR / 'metrics'}")

if __name__ == "__main__":
    main()