"""
Generate feature importance across different models.
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.preprocessor import Preprocessor
from src.models.random_forest import RandomForestModel
from src.models.decision_tree import DecisionTreeModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.adaboost import AdaBoostModel
from config import *

sns.set_style('whitegrid')

def get_feature_importance_from_models(wine_type):
    X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)

    importance_data = {}

    rf_params = {'n_estimators': 100, 'max_depth': 20 if wine_type == 'red' else None, 
                 'random_state': RANDOM_SEED}
    rf_model = RandomForestModel(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_importance = rf_model.get_feature_importance()
    importance_data['Random Forest'] = rf_importance.set_index('feature')['importance']

    dt_model = DecisionTreeModel(max_depth=None, criterion='entropy', random_state=RANDOM_SEED)
    dt_model.fit(X_train, y_train)
    dt_importance = dt_model.get_feature_importance()
    importance_data['Decision Tree'] = dt_importance.set_index('feature')['importance']

    ada_model = AdaBoostModel(n_estimators=100, learning_rate=1.0, 
                              base_estimator_max_depth=1, random_state=RANDOM_SEED)
    ada_model.fit(X_train, y_train)
    ada_importance = ada_model.get_feature_importance()
    importance_data['AdaBoost'] = ada_importance.set_index('feature')['importance']
    
    lr_model = LogisticRegressionModel(C=1.0, random_state=RANDOM_SEED)
    lr_model.fit(X_train, y_train)
    lr_importance = lr_model.get_feature_importance()
    importance_data['Logistic Regression'] = lr_importance.set_index(lr_importance.index)['avg_importance']

    importance_df = pd.DataFrame(importance_data)

    importance_df_norm = importance_df.div(importance_df.sum(axis=0), axis=1)

    return importance_df_norm

def plot_feature_importance_comparison(importance_df, wine_type, save_path):
    importance_df['mean'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('mean', ascending=True)
    importance_df = importance_df.drop('mean', axis=1)

    fig, ax = plt.subplots(figsize=(14, 10))

    importance_df.plot(kind='barh', ax=ax, width=0.8)

    ax.set_xlabel('Normalized Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance Comparison - {wine_type.capitalize()} Wine\nAcross Different Models',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Model', fontsize=10, title_fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap_comparison(red_importance, white_importance, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    for idx, (wine_type, importance_df) in enumerate([('red', red_importance), 
                                                        ('white', white_importance)]):
        ax = axes[idx]
        
        importance_df['mean'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('mean', ascending=False)
        importance_df = importance_df.drop('mean', axis=1)
        
        sns.heatmap(importance_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Normalized Importance'})
        
        ax.set_title(f'{wine_type.capitalize()} Wine\nFeature Importance Heatmap',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_importance_table(red_importance, white_importance, save_path):
    red_importance['wine_type'] = 'red'
    white_importance['wine_type'] = 'white'
    
    red_importance = red_importance.reset_index().rename(columns={'index': 'feature'})
    white_importance = white_importance.reset_index().rename(columns={'index': 'feature'})
    
    combined = pd.concat([red_importance, white_importance], ignore_index=True)
    
    cols = ['wine_type', 'feature'] + [col for col in combined.columns 
                                        if col not in ['wine_type', 'feature']]
    combined = combined[cols]
    
    combined.to_csv(save_path, index=False)

def main():
    output_dir = RESULTS_DIR / 'figures' / 'feature_importance'
    output_dir.mkdir(parents=True, exist_ok=True)

    red_importance = get_feature_importance_from_models('red')
    white_importance = get_feature_importance_from_models('white')

    plot_feature_importance_comparison(red_importance.copy(), 'red',
                                       output_dir / 'red_feature_importance_comparison.png')
    plot_feature_importance_comparison(white_importance.copy(), 'white',
                                       output_dir / 'white_feature_importance_comparison.png')
    
    plot_heatmap_comparison(red_importance.copy(), white_importance.copy(),
                            output_dir / 'feature_importance_heatmap.png')
    
    save_importance_table(red_importance.copy(), white_importance.copy(),
                          output_dir / 'feature_importance_data.csv')
    
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    main()