# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 3: CREDIT RISK

# 1. Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 2. Generate Data
# Let's assume we have two features: credit_score and annual_income, and the target is credit_risk (0 for low risk, 1 for high risk)
np.random.seed(0)
data_size = 1000
credit_scores = np.random.normal(600, 100, data_size)
annual_incomes = np.random.normal(50000, 15000, data_size)
credit_risks = (credit_scores < 580) | (annual_incomes < 30000)  # Simplified risk criteria

# Create a DataFrame
df = pd.DataFrame({
    'credit_score': credit_scores,
    'annual_income': annual_incomes,
    'credit_risk': credit_risks.astype(int)
})

# 3. Preprocess Data
# Split data into features and target
X = df[['credit_score', 'annual_income']]
y = df['credit_risk']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train a Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)