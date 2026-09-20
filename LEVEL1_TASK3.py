import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read dataset
df = pd.read_csv('1.iris.csv')

# 2. Create basic plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1), sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.subplot(1, 2, 2), df['species'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Species Count')
plt.show()
