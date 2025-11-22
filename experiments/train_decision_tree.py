"""
Train and evaluate Decision Tree model.
"""

import sys
sys.path.append('..')

from src.data.preprocessor import Preprocessor
from src.models.decision_tree import DecisionTreeModel
from src.evaluation.metrics import (
    evaluate_model, print_evaluation,
    print_confusion_matrix, print_per_class_metrics
)
from src.utils.experiment_tracker import ExperimentTracker
from config import *

def train_decision_tree(wine_type: str, max_depth: int = None,
                        min_samples_split: int = 2, min_samples_leaf: int = 1):
    print(f'Training Decision Tree for {wine_type.upper()} wines')
    print(f'Hyperparameters: max_depth={max_depth}, '
          f'min_samples_split={min_samples_split}, '
          f'min_samples_leaf={min_samples_leaf}')
    
    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    model = DecisionTreeModel(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion='entropy',
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
    max_depths = [None, 10, 5]

    for wine_type in wine_types:
        for max_depth in max_depths:
            model, results = train_decision_tree(
                wine_type, 
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1
            )

    print(f"\nResults saved to: {RESULTS_DIR / 'metrics'}")

if __name__ == "__main__":
    main()