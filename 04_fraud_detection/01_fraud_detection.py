# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 4: FRAUD DETECTION

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data
data_size = 1000
transaction_types = np.random.choice(['type1', 'type2', 'type3'], size=data_size)
transaction_amounts = np.random.exponential(scale=200, size=data_size)
age_of_account_days = np.random.normal(loc=365, scale=100, size=data_size)

# Conditional probabilities for being fraudulent
fraudulent = []
for i in range(data_size):
    # Higher chance of fraud for type1 with high transaction amount and low account age
    if transaction_types[i] == 'type1' and transaction_amounts[i] > 100 and age_of_account_days[i] < 365:
        fraudulent.append(np.random.choice([0, 1], p=[0.1, 0.9]))  # 90% chance of being fraudulent
    else:
        fraudulent.append(np.random.choice([0, 1], p=[0.99, 0.01]))  # 1% chance as normal

df = pd.DataFrame({
    'transaction_amount': transaction_amounts,
    'transaction_type': transaction_types,
    'age_of_account_days': age_of_account_days,
    'fraudulent': fraudulent
})

df_untransformed = df.copy()

# Encode categorical data
df['transaction_type'] = LabelEncoder().fit_transform(df['transaction_type'])

# Prepare data for training
X = df.drop('fraudulent', axis=1)
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions
predictions_df = pd.DataFrame(model.predict(X_test), columns=['predict_class'])
predictions_proba_df = pd.DataFrame(model.predict_proba(X_test), columns=["p0", "p1"])

# Test Set Evaluation
fraud_scoring_df = pd.concat([
    df_untransformed.iloc[X_test.index.values].reset_index(), predictions_df,
    predictions_proba_df
], axis=1) \
    .set_index("index")
    
fraud_scoring_df.sort_values('p1', ascending=False)

print(classification_report(fraud_scoring_df['fraudulent'], fraud_scoring_df['predict_class']))
