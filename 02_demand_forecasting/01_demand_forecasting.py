# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 2: DEMAND FORECASTING

# libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set the seed for reproducibility
np.random.seed(123)

# Generate the date sequence
date_seq = pd.date_range(start="2020-01-01", end="2021-12-31", freq='D')

# Create the dataframe
data = pd.DataFrame({
    'date': date_seq,
    'demand': np.round(100 + np.sin(np.arange(len(date_seq)) / 20) * 50 + np.random.normal(0, 10, len(date_seq)))
})

# Convert date column to datetime and sort the data
data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True)

# Split the data into training and testing sets
split_date = pd.Timestamp('2021-06-01')
train_data = data[data['date'] < split_date]
test_data = data[data['date'] >= split_date]

def create_features(df):
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Fourier features for capturing seasonality
    for k in range(1, 3):  # K=2 fourier pairs
        df[f'sin{k}'] = np.sin(df['dayofyear'] * (2. * np.pi * k / 365.25))
        df[f'cos{k}'] = np.cos(df['dayofyear'] * (2. * np.pi * k / 365.25))
    return df

train_features = create_features(train_data.copy())
test_features = create_features(test_data.copy())

# Define the features and the target
feature_columns = ['dayofyear', 'dayofweek', 'month', 'sin1', 'cos1', 'sin2', 'cos2']
target_column = 'demand'

# Model Pipeline
pipeline = Pipeline(steps=[
    ('encoder', ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['dayofweek', 'month'])],
        remainder='passthrough')),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror'))
])

# Fit the model
pipeline.fit(train_features[feature_columns], train_features[target_column])

# Predict using the model
train_preds = pipeline.predict(train_features[feature_columns])
test_preds = pipeline.predict(test_features[feature_columns])

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(train_features[target_column], train_preds))
test_rmse = np.sqrt(mean_squared_error(test_features[target_column], test_preds))

print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Create future dates for forecasting
future_dates = pd.date_range(start=test_data['date'].max() + timedelta(days=1), periods=180, freq='D')
future_data = pd.DataFrame({'date': future_dates})
future_features = create_features(future_data.copy())

# Forecast future demand
future_preds = pipeline.predict(future_features[feature_columns])

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train_data['date'], train_data['demand'], label='Train Data')
plt.plot(test_data['date'], test_data['demand'], label='Test Data')
plt.plot(future_dates, future_preds, label='Forecast', color='red')
plt.legend()
plt.title('Demand Forecast')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()
