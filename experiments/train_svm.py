"""
Train and evaluate Support Vector Machine Model.
"""

import sys
sys.path.append('..')

from src.data.preprocessor import Preprocessor
from src.models.svm import SVMModel
from src.evaluation.metrics import (
    evaluate_model, print_evaluation,
    print_confusion_matrix, print_per_class_metrics
)
from src.utils.experiment_tracker import ExperimentTracker
from config import *

def train_svm(wine_type: str, C: float = 1.0, kernel: str = 'rbf',
              gamma: str = 'scale'):
    print(f'Training SVM for {wine_type.upper()} wines')
    print(f'Hyperparameters: C={C}, kernel={kernel}, gamma={gamma}')

    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    model = SVMModel(
        C=C,
        kernel=kernel,
        gamma=gamma,
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

    if kernel == 'linear':
        print('\nTop 5 Most Important Features:')
        feature_importance = model.get_feature_importance()
        print(feature_importance.head())

    return model, results

def main():
    wine_types = ['red', 'white']
    C_values = [0.1, 1.0, 10.0]

    kernel = 'rbf'

    for wine_type in wine_types:
        for C in C_values:
            model, results = train_svm(
                wine_type,
                C=C,
                kernel=kernel,
                gamma='scale'
            )

    print(f"\nResults saved to: {RESULTS_DIR / 'metrics'}")

if __name__ == "__main__":
    main()