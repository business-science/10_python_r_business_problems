# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 4: FRAUD DETECTION

# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
data_size = 1000
data = {
    'transaction_amount': np.random.exponential(scale=200, size=data_size),
    'transaction_type': np.random.choice(['type1', 'type2', 'type3'], size=data_size),
    'age_of_account_days': np.random.normal(loc=365, scale=100, size=data_size),
    'fraudulent': np.random.choice([0, 1], size=data_size, p=[0.95, 0.05])
}
df = pd.DataFrame(data)

# Encode categorical data
df['transaction_type'] = LabelEncoder().fit_transform(df['transaction_type'])

# Scale numerical data
scaler = StandardScaler()
df['transaction_amount'] = scaler.fit_transform(df[['transaction_amount']])
df['age_of_account_days'] = scaler.fit_transform(df[['age_of_account_days']])

# Prepare data for training
X = df.drop('fraudulent', axis=1)
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict_proba(X_test)
predictions_df = pd.DataFrame(predictions, columns=["p0", "p1"])

pd.concat([X_test.reset_index(), predictions_df, y_test.reset_index().drop('index', axis = 1)], axis=1) \
    .set_index("index") \
    .sort_values('p1', ascending=False)


