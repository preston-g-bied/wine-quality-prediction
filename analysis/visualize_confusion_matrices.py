"""
Generate confusion matrix visualizations for best performing models
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.preprocessor import Preprocessor
from src.models.random_forest import RandomForestModel
from src.models.knn import KNNModel
from config import *

sns.set_style('whitegrid')

def plot_confusion_matrix(y_true, y_pred, wine_type, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = ['Low', 'Medium', 'High']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
    
    ax.set_title(f'{model_name} - {wine_type.capitalize()} Wine\nConfusion Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Quality Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Quality Class', fontsize=12, fontweight='bold')
    
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, 1.08, f'Accuracy: {accuracy:.1%}', 
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():    
    output_dir = RESULTS_DIR / 'figures' / 'confusion_matrices'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_models = {
        'red': {
            'name': 'Random Forest',
            'model': RandomForestModel(n_estimators=100, max_depth=20, random_state=RANDOM_SEED)
        },
        'white': {
            'name': 'Random Forest', 
            'model': RandomForestModel(n_estimators=200, max_depth=None, random_state=RANDOM_SEED)
        }
    }
    
    knn_models = {
        'red': {
            'name': 'KNN (Distance-Weighted)',
            'model': KNNModel(n_neighbors=3, weights='distance')
        },
        'white': {
            'name': 'KNN (Distance-Weighted)',
            'model': KNNModel(n_neighbors=11, weights='distance')
        }
    }
    
    for wine_type in ['red', 'white']:
        
        X_train, X_test, y_train, y_test = Preprocessor.load_processed_data(wine_type)
        
        rf_model = best_models[wine_type]['model']
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        
        save_path = output_dir / f'{wine_type}_random_forest_confusion_matrix.png'
        plot_confusion_matrix(y_test, rf_preds, wine_type, 
                            best_models[wine_type]['name'], save_path)
        
        knn_model = knn_models[wine_type]['model']
        knn_model.fit(X_train, y_train)
        knn_preds = knn_model.predict(X_test)
        
        save_path = output_dir / f'{wine_type}_knn_confusion_matrix.png'
        plot_confusion_matrix(y_test, knn_preds, wine_type,
                            knn_models[wine_type]['name'], save_path)
    
    print(f"All confusion matrices saved to: {output_dir}")

if __name__ == "__main__":
    main()