"""
Track experiment results.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from config import *

class ExperimentTracker:
    def __init__(self):
        self.results_file = RESULTS_DIR / 'metrics' / 'experiment_results.csv'
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, experiment_data: Dict[str, Any]) -> None:
        """Log experiment results to CSV"""
        experiment_data['timestamp'] = datetime.now().isoformat()

        df = pd.DataFrame([experiment_data])

        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)