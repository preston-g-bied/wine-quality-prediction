"""
Configuration for the whole project.
"""

from pathlib import Path

# set random seed
RANDOM_SEED = 49

# set filepaths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

# wine color paths
RED_WINE_PATH = RAW_DATA_DIR / "winequality-red.csv"
WHITE_WINE_PATH = RAW_DATA_DIR / "winequality-white.csv"

# quality classification bins
QUALITY_BINS = {
    'low': (0, 5),
    'medium': (6, 6),
    'high': (7, 10)
}
BIN_EDGES = [0, 6, 7, 11]  # 0-5, 6, 7-10
BIN_LABELS = ['low', 'medium', 'high']

# train/test split configuration
TEST_SIZE = 0.3

# feature names
FEATURE_COLUMNS = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

# target
TARGET_COLUMN = 'quality'