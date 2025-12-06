"""
Regression vs Classification Comparison
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from config import *

output_dir = RESULTS_DIR / 'figures' / 'regression_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

class RegressionModel:
    """Random Forest Regressor for wine quality prediction"""

    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_feature_importance(self):
        feature_names = self.model.feature_names_in_
        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        return importance_df.sort_values('importance', ascending=False)
    
def load_original_quality_data(wine_type):
    data_loader = DataLoader()

    df = data_loader.load_wine_data(wine_type)

    X = df.drop(columns=['quality'])
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_regression(y_true, y_pred, wine_type):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    differences = np.abs(y_true - y_pred)

    exact_accuracy = np.mean(differences < 0.5)
    tolerance_1 = np.mean(differences <= 1.0)
    tolerance_1_5 = np.mean(differences <= 1.5)

    results = {
        'wine_type': wine_type,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'exact_accuracy': exact_accuracy,
        'tolerance_1': tolerance_1,
        'tolerance_1_5': tolerance_1_5
    }

    return results

def plot_regression_results(y_true, y_pred, wine_type, results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_true, y_pred, alpha=0.5, c=np.abs(y_true - y_pred), 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Quality Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Quality Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'{wine_type.capitalize()} Wine - Regression Performance\nMAE: {results["mae"]:.3f}, R²: {results["r2"]:.3f}',
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Absolute Error')
    
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Quality Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{wine_type.capitalize()} Wine - Residuals Plot',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    errors = np.abs(y_true - y_pred)
    ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(results['mae'], color='r', linestyle='--', lw=2, label=f'MAE: {results["mae"]:.3f}')
    ax3.set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title(f'{wine_type.capitalize()} Wine - Error Distribution',
                 fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    bins = np.arange(2.5, 10.5, 0.5)
    ax4.hist(y_true, bins=bins, alpha=0.5, label='Actual', edgecolor='black', color='blue')
    ax4.hist(y_pred, bins=bins, alpha=0.5, label='Predicted', edgecolor='black', color='orange')
    ax4.set_xlabel('Quality Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title(f'{wine_type.capitalize()} Wine - Distribution Comparison',
                 fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{wine_type}_regression_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{wine_type}_regression_analysis.png'}")
    plt.close()

def compare_regression_vs_classification():
    classification_results = pd.read_csv(RESULTS_DIR / 'summary_results.csv')

    comparison_data = []

    for wine_type in ['red', 'white']:
        class_best = classification_results[
            (classification_results['wine_type'] == wine_type) &
            (classification_results['model_type'] == 'Random Forest')
        ].iloc[0]

        comparison_data.append({
            'wine_type': wine_type,
            'approach': 'Classification (3-class)',
            'primary_metric': class_best['accuracy'],
            'metric_name': 'Accuracy',
            'precision': class_best['precision'],
            'recall': class_best['recall'],
            'f1': class_best['f1']
        })

    return pd.DataFrame(comparison_data)

def create_comparison_visualization(regression_results, classification_results):
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    wine_types = ['red', 'white']
    
    for idx, wine_type in enumerate(wine_types):
        ax = axes[0, idx]
        
        reg_result = regression_results[regression_results['wine_type'] == wine_type].iloc[0]
        class_result = classification_results[classification_results['wine_type'] == wine_type].iloc[0]
        
        approaches = ['Regression\n(MAE)', 'Classification\n(3-class Accuracy)']
        values = [reg_result['mae'], class_result['primary_metric']]
        colors = ['steelblue', 'coral']
        
        bars = ax.bar(approaches, values, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{wine_type.capitalize()} Wine - Approach Comparison\n(Lower MAE vs Higher Accuracy is better)',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for idx, wine_type in enumerate(wine_types):
        ax = axes[1, idx]
        
        reg_result = regression_results[regression_results['wine_type'] == wine_type].iloc[0]
        class_result = classification_results[classification_results['wine_type'] == wine_type].iloc[0]
        
        tolerances = ['Exact\n(±0.5)', 'Within 1', 'Within 1.5', '3-Class\nAccuracy']
        reg_values = [
            reg_result['exact_accuracy'],
            reg_result['tolerance_1'],
            reg_result['tolerance_1_5'],
            0
        ]
        class_values = [
            0,
            0,
            0,
            class_result['primary_metric']
        ]
        
        x = np.arange(len(tolerances))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, reg_values, width, label='Regression', 
                      color='steelblue', edgecolor='black', alpha=0.7)
        bars2 = ax.bar(x + width/2, class_values, width, label='Classification',
                      color='coral', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Tolerance Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{wine_type.capitalize()} Wine - Tolerance-based Accuracy',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tolerances, fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_vs_classification_comparison.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'regression_vs_classification_comparison.png'}")
    plt.close()

def main():
    regression_results = []

    for wine_type in ['red', 'white']:
        print(f"Training Regression Model for {wine_type.upper()} Wine")

        X_train, X_test, y_train, y_test = load_original_quality_data(wine_type)

        max_depth = 20 if wine_type == 'red' else None
        model = RegressionModel(
            n_estimators=100,
            max_depth=max_depth,
            random_state=RANDOM_SEED
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results = evaluate_regression(y_test, y_pred, wine_type)
        regression_results.append(results)

        print(f"\nRegression Results:")
        print(f"  MAE:  {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  R²:   {results['r2']:.4f}")
        print(f"\nTolerance-based Accuracies:")
        print(f"  Exact (±0.5):    {results['exact_accuracy']:.4f}")
        print(f"  Within 1 point:  {results['tolerance_1']:.4f}")
        print(f"  Within 1.5:      {results['tolerance_1_5']:.4f}")

        plot_regression_results(y_test, y_pred, wine_type, results)

        feature_imp = model.get_feature_importance()
        print(f"\nTop 5 Most Important Features:")
        print(feature_imp.head().to_string(index=False))

    regression_df = pd.DataFrame(regression_results)
    regression_df.to_csv(output_dir / 'regression_results.csv', index=False)
    print(f"\nSaved regression results to: {output_dir / 'regression_results.csv'}")

    print("\nCOMPARING REGRESSION VS CLASSIFICATION")

    classification_df = compare_regression_vs_classification()

    comparison_table = pd.concat([
        regression_df[['wine_type', 'mae', 'rmse', 'r2', 'exact_accuracy', 
                      'tolerance_1', 'tolerance_1_5']],
        classification_df[['wine_type', 'primary_metric', 'precision', 'recall', 'f1']]
    ], axis=1)

    print("\nComparison Summary:")
    print(comparison_table.to_string(index=False))

    create_comparison_visualization(regression_df, classification_df)

    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main()