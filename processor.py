import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

def clean_data(df):
    if df.empty:
        return df
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['date', 'amount'])
    df = df.sort_values('date')
    return df

def detect_anomalies(df, contamination=0.05):
    if len(df) < 10:
        df['is_anomaly'] = False
        return df
        
    model = IsolationForest(contamination=contamination, random_state=42)
    amounts = df['amount'].values.reshape(-1, 1)
    # Isolation forest returns -1 for anomalies, 1 for normal
    df['is_anomaly'] = model.fit_predict(amounts) == -1
    return df

def predict_cashflow(df):
    if df.empty:
        return None, "No data available."
        
    df['Month_Index'] = (df['date'].dt.year - df['date'].dt.year.min()) * 12 + df['date'].dt.month
    monthly_spend = df.groupby('Month_Index')['amount'].sum().reset_index()
    
    if len(monthly_spend) < 2:
        return None, "Need at least 2 months of historical data to predict."
        
    X = monthly_spend[['Month_Index']]
    y = monthly_spend['amount']
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_month = monthly_spend['Month_Index'].max() + 1
    prediction = model.predict([[next_month]])[0]
    
    return max(0, prediction), "Success"
