Python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read dataset
df = pd.read_csv('2.Stock Prices Data Set (1).csv')

# 2. Show statistics and correlation
print(df.describe())
print(numeric_df := df.select_dtypes(include=['float64', 'int64']).corr())

# 3. Draw charts
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1), sns.histplot(df['close'], kde=True)
plt.subplot(1, 2, 2), sns.heatmap(numeric_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

print("Level 1 - Task 2 completed.")
