import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# 1. Read dataset
df = pd.read_csv('3.Sentiment dataset.csv').dropna(subset=['Text'])

# 2. Calculate sentiment polarity
df['polarity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# 3. Show statistics and plot histogram
print(df['polarity'].describe())
plt.figure(figsize=(6, 4))
df['polarity'].hist(bins=20, color='purple', edgecolor='black')
plt.title('Sentiment Polarity Distribution')
plt.show()

print("Level 3 - Task 3 completed.")
