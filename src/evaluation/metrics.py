"""
Evaluation metrics for models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from typing import Dict, Any
import config

def evaluate_model(y_true: pd.Series, y_pred: pd.Series, wine_type: str, model_name: str = "Model") -> Dict[str, Any]:
    """Return dictionary of all evaluation metrics"""
    results = {
        'model_name': model_name,
        'wine_type': wine_type,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return results

def get_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

def print_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    cm = get_confusion_matrix(y_true, y_pred)
    labels = sorted(y_true.unique())
    
    print("\nCONFUSION MATRIX:")
    
    # Print header
    header = "Actual/Pred |" + "".join([f" {label:>8}" for label in labels])
    print(header)
    print("-"*50)
    
    # Print rows
    for i, label in enumerate(labels):
        row = f"    {label:>7} |" + "".join([f" {cm[i][j]:>8}" for j in range(len(labels))])
        print(row)

def print_per_class_metrics(y_true: pd.Series, y_pred: pd.Series) -> None:
    print("\nPER-CLASS METRICS:")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

def print_evaluation(results: Dict[str, Any]) -> None:
    """Print evaluation results"""
    print(f"\n{'='*50}")
    print(f"Results for {results['model_name']} on {results['wine_type']} wines")
    print(f"{'='*50}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"{'='*50}\n")