import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Read dataset
df = pd.read_csv('4.house Prediction Data Set.csv', header=None, delim_whitespace=True)
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# 2. Train-test split and fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

print("Level 2 - Task 1 completed.")
