import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
import os

class SimpleDB:
    def __init__(self):
        # Simple connection string
        server = os.getenv('SQL_SERVER', 'localhost') 
        database = os.getenv('SQL_DATABASE', 'delay_prediction') 
        username = os.getenv('SQL_USERNAME', 'your_username') 
        password = os.getenv('SQL_PASSWORD', 'your_password') 
        
        connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server" 
        self.engine = create_engine(connection_string)
    
    def get_data(self, hours_back=24): 
        """Get recent data for predictions"""
        query = f"""
        SELECT datetime, time_taken, CPU, RAM, sc_status, is_error
        FROM system_metrics 
        WHERE datetime >= DATEADD(HOUR, -{hours_back}, GETDATE())
        ORDER BY datetime
        """
        return pd.read_sql(query, self.engine)
    
    def save_predictions(self, predictions_df):
        """Save predictions to database"""
        predictions_df.to_sql('predictions', self.engine, if_exists='append', index=False)
