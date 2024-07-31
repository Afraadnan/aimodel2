import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("sample3.csv")
df.columns = df.columns.str.strip()

# Impute missing values in TargetUnits, SaleUnits, and AchSale
si = SimpleImputer(missing_values=np.nan, strategy="median")
df[['TargetUnits', 'SaleUnits', 'AchSale']] = si.fit_transform(df[['TargetUnits', 'SaleUnits', 'AchSale']]) 

# Define features and target for training the model
X = df[['TargetUnits', 'SaleUnits']]
y = df['AchSale']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Scale features
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train a Decision Tree Regressor
clf = DecisionTreeRegressor(max_depth=50)
clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test_scaled)
again= clf.predict[[17,23]]
print (again)

# Print lengths
print("Length of y_test:", len(y_test))
print("Length of predictions:", len(predictions))

# Plot predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Sales Predictor')
plt.show()

# Define MASE function
def mase(y_test, estimated, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train, axis=0)).sum() / (n - 1)
    errors = np.abs(y_test - estimated)
    return errors.mean() / d

# Calculate MASE
mase_value = mase(y_test, predictions, y_train)
print("MASE:", mase_value)

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
mase_scores = []

for train_index, test_index in kfold.split(X):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale features for each fold
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    clf.fit(x_train_scaled, y_train)
    estimated = clf.predict(x_test_scaled)

    mse = mean_squared_error(y_test, estimated)
    mse_scores.append(mse)

    mase_value = mase(y_test, estimated, y_train)
    mase_scores.append(mase_value)

print("Mean MSE:", np.mean(mse_scores))
print("Mean MASE:", np.mean(mase_scores))
