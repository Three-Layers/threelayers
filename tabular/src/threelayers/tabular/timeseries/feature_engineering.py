import pandas as pd
import numpy as np

def time_features(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    time_periods = [('hour', 24, df.index.hour), ('month', 30, df.index.month), ('year', 365, df.index.year)]
    
    for time, period, series in time_periods:
        df[f'{time}_sin'] = np.sin(2 * np.pi * series / period)
        df[f'{time}_cos'] = np.cos(2 * np.pi * series / period)
        
    return df

def lag_features(_df: pd.DataFrame, columns: list = None, lag_hours: list = [1]) -> pd.DataFrame:
    df = _df.copy()
    if not columns: columns = df.columns
    for col in columns:
        if df[col].dtype in [int, float]:
            for hour in lag_hours:
                # Kalau time series-nya per jam
                # df[f'{col}_lag_{hour}h'] = df[col].shift(freq=pd.Timedelta(hours=hour))
                df[f'{col}_lag_{hour}h'] = df[col].shift(hour)
    return df

def diff_features(_df: pd.DataFrame, columns: list = None, diff_hours: list = [1]) -> pd.DataFrame:
    df = _df.copy()
    if not columns: columns = df.columns
    for col in columns:
        if df[col].dtype in [int, float]:
            for hour in diff_hours:
                df[f'{col}_diff_{hour}h'] = df[col] - df[col].shift(hour)
    return df