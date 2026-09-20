import pandas as pd
import matplotlib.pyplot as plt

# 1. Read dataset and parse dates
df = pd.read_csv('2.Stock Prices Data Set (1).csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 2. Moving average smoothing
df['rolling_mean'] = df['close'].rolling(window=30).mean()

# 3. Plot time series
plt.figure(figsize=(10, 4))
plt.plot(df['close'], label='Close Price', alpha=0.5)
plt.plot(df['rolling_mean'], label='30-Day Moving Average', color='red')
plt.legend(), plt.show()

print("Level 2-Task 2 completed.")
