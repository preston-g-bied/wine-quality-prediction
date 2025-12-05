"""
Train and evaluate K-Nearest Neighbors model.
"""

import sys
sys.path.append('..')

from src.data.preprocessor import Preprocessor
from src.models.knn import KNNModel
from src.evaluation.metrics import (
    evaluate_model, print_evaluation,
    print_confusion_matrix, print_per_class_metrics
)
from src.utils.experiment_tracker import ExperimentTracker
from config import *

def train_knn(wine_type: str, n_neighbors: int = 5, weights: str = 'uniform'):
    print(f'Training KNN for {wine_type.upper()} wines')
    print(f'Hyperparameters: k={n_neighbors}, weights={weights}')

    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    model = KNNModel(
        n_neighbors=n_neighbors,
        weights=weights,
        metric='minkowski',
        p=2
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

    return model, results

def main():
    wine_types = ['red', 'white']

    k_values = [3, 5, 7, 9, 11]
    weight_schemes = ['uniform', 'distance']

    for wine_type in wine_types:
        for k in k_values:
            for weight_scheme in weight_schemes:
                model, results = train_knn(
                    wine_type,
                    n_neighbors=k,
                    weights=weight_scheme
                )

    print(f"\nResults saved to: {RESULTS_DIR / 'metrics'}")

if __name__ == "__main__":
    main()