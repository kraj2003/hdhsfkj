# src/data_validator.py - Simple Data Validation
import pandas as pd
import numpy as np
from typing import List, Dict

def validate_raw_data(df: pd.DataFrame) -> Dict:
    """Simple validation for raw data"""
    issues = []
    
    # Check required columns
    required_cols = ['timestamp', 'response_time', 'CPU', 'RAM']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check data ranges
    if 'CPU' in df.columns:
        invalid_cpu = ((df['CPU'] < 0) | (df['CPU'] > 100)).sum()
        if invalid_cpu > 0:
            issues.append(f"Invalid CPU values: {invalid_cpu} records")
    
    if 'RAM' in df.columns:
        invalid_ram = ((df['RAM'] < 0) | (df['RAM'] > 100)).sum()
        if invalid_ram > 0:
            issues.append(f"Invalid RAM values: {invalid_ram} records")
    
    if 'response_time' in df.columns:
        invalid_response = (df['response_time'] < 0).sum()
        if invalid_response > 0:
            issues.append(f"Negative response times: {invalid_response} records")
    
    # Check for too many nulls
    null_pct = df.isnull().sum() / len(df)
    high_null_cols = null_pct[null_pct > 0.1].index.tolist()
    if high_null_cols:
        issues.append(f"High null percentage in: {high_null_cols}")
    
    return {
        'total_records': len(df),
        'issues': issues,
        'is_valid': len(issues) == 0,
        'null_percentages': null_pct.to_dict()
    }

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simple data cleaning"""
    df = df.copy()
    
    # Fix obvious issues
    if 'CPU' in df.columns:
        df['CPU'] = df['CPU'].clip(0, 100).fillna(0)
    
    if 'RAM' in df.columns:
        df['RAM'] = df['RAM'].fillna(df['RAM'].median()).clip(0, 100)
    
    if 'response_time' in df.columns:
        df['response_time'] = df['response_time'].clip(0, None).fillna(0)

    
    return df