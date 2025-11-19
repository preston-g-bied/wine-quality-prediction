"""
Train and evaluate Logistic Regression model.
"""

import sys
sys.path.append('..')

from src.data.preprocessor import Preprocessor
from src.models.logistic_regression import LogisticRegressionModel
from src.evaluation.metrics import evaluate_model, print_evaluation
from src.utils.experiment_tracker import ExperimentTracker
from config import *

def train_logistic_regression(wine_type: str, C: float = 1.0):
    print(f"Train Logistic Regression for {wine_type.upper()} wines")

    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    model = LogisticRegressionModel(
        C=C,
        max_iter=1000,
        random_state=RANDOM_SEED
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = evaluate_model(y_test, predictions, wine_type, model.get_name())
    print_evaluation(results)

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
    
    for wine_type in wine_types:
        model, results = train_logistic_regression(wine_type, C=1.0)

    print(f"\nResults saved to: {RESULTS_DIR / 'metrics'}")

if __name__ == "__main__":
    main()