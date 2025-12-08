# Wine Quality Prediction

CS 613 - Machine Learnng Final Project
Preston Bied

## Overview

This project predicts wine quality ratings based on physicochemical properties using various machine learning algorithms. I'm using the UCI Wine Quality dataset and comparing six different classification approaches to see which works best for predicting whether a wine is low, medium, or high quality.

## Dataset

The data comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). It includes:
- 1,599 red wine samples
- 4,898 white wine samples
- 11 physicochemical features (alcohol content, acidity, pH, etc.)
- Quality ratings from 0-10 (given by wine experts)

I converted the numeric ratings into three categories:
- **Low:** 0-5
- **Medium:** 6
- **High:** 7-10

The red and white wine datasets are kept separate throughout the analysis since they have pretty different characteristics.

## Project Structure


```
wine-quality-prediction/
├── data/
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Preprocessed/standardized data
│   └── splits/                 # Train/test splits
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Load and split data
│   │   └── preprocessor.py     # Standardization
│   ├── models/
│   │   ├── base_model.py       # Abstract base class
│   │   ├── baseline.py
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── decision_tree.py
│   │   ├── knn.py
│   │   ├── random_forest.py
│   │   └── adaboost.py
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy, precision, recall, F1
│   └── utils/
│       └── experiment_tracker.py
├── experiments/
│   ├── train_baseline.py
│   ├── train_logistic_regression.py
│   ├── train_svm.py
│   ├── train_decision_tree.py
│   ├── train_knn.py
│   ├── train_random_forest.py
│   └── train_adaboost.py
├── analysis/
│   ├── compare_feature_importance.py
│   ├── visualize_confusion_matrices.py
│   └── regression_comparison.py
├── results/
│   ├── metrics/                # CSV files with all results
│   └── figures/                # Generated plots
├── config.py
└── README.md
```

## Setup

### Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

### Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- Get `winequality-red.csv` and `winequality-white.csv` from UCI
- Place them in `data/raw/`

## How to Reproduce Results

I've organize the code so you can run everything step-by-step.

### 1. Data Preparation

First, create the train/test splits:

```bash
python3 -m src.data.data_loader
```

This creates stratified 70/30 splits and saves them in `data/splits`.

Next, preprocess the data (standardization):

```bash
python3 -m src.data.preprocessor
```

This saves standardized data to `data/processed/`.

You can also view my EDA notebook at `notebooks/exploratory_analysis.ipynb`.

### 2. Train Models

Train each model by running its script in the `experiments/` directory:

```bash
python3 -m experiments.train_baseline

python3 -m experiments.train_logistic_regression
python3 -m experiments.train_svm
python3 -m experiments.train_decision_tree
python3 -m experiments.train_knn

python3 -m experiments.train_random_forest
python3 -m experiments.train_adaboost
```

Each script trains the model on both red and white wine, runs hyperparameter tuning, and logs results to CSV files in `results/metrics/`.

### 3. Generate Analysis

You can view my comprehensive results analysis at `notebooks/results_analysis.ipynb`.

Run the analysis scripts to create visualizations:

```bash
python3 -m analysis.compare_feature_importance

python3 -m analysis.visualize_confusion_matrices

python3 -m analysis.regression_comparison
```

These save figures to `results/figures/`.

### Key Results

**Best Models:**
- **Red Wine:** Random Forest (n_estimators=100, max_depth=20) -> 75.8% accuracy
- **White Wine:** Random Forest (n_estimators=200, max_depth=None) -> 70.3% accuracy

**Algorithm Rankings (by accuracy):**
1. Random Forest
2. K-Nearest Neighbors
3. Support Vector Machine
4. AdaBoost
5. Decision Tree
6. Logistic Regression

**Key Findings:**
- Ensemble methods significantly outperform individual algorithms
- Distance-weighted KNN performs surprisingly well
- Different features matter for red vs. white wines
- Alcohol content and sulphates are consistently important accross both wine types
- Classification approach works better than regression

## Notes

- All experiments use a fixed random seed (49) for reproducibility
- Results are logged to CSV files with timestamps
- The code ses scikit-learn for all algorithms
- A small amount of hyperparameter tuning was done for each algorithm

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553.