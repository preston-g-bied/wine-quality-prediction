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
    def __init__(self, model_name: str, wine_type: str):
        self.results_file = RESULTS_DIR / 'metrics' / f'experiment_results_{model_name}_{wine_type}.csv'
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.wine_type = wine_type

    def log_experiment(self, experiment_data: Dict[str, Any]) -> None:
        """Log experiment results to CSV"""
        experiment_data['timestamp'] = datetime.now().isoformat()

        df = pd.DataFrame([experiment_data])

        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

        print(f"\nLOGGED experiment for {self.model_name} on {self.wine_type} wines\n")