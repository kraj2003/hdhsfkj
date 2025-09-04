import pandas as pd
from src.database import SimpleDB
import os

class DataManager:
    def __init__(self):
        self.use_sql = os.getenv('USE_SQL', 'false').lower() == 'true'
        if self.use_sql:
            self.db = SimpleDB()
    
    def load_data(self):
        """Load data from CSV or SQL"""
        if self.use_sql:
            return self.db.get_data()
        else:
            return pd.read_csv('./data/final_dataset.csv')
    
    def save_results(self, results_df):
        """Save results to CSV or SQL"""
        if self.use_sql:
            self.db.save_predictions(results_df)
        else:
            results_df.to_csv('predictions.csv', index=False)
