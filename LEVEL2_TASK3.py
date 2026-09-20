import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Prepare data
df = pd.read_csv('1.iris.csv')
X = StandardScaler().fit_transform(df.select_dtypes(include=['float64']))

# 2. K-Means clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
df['cluster'] = kmeans.labels_

# 3. Visualize clusters
plt.figure(figsize=(6, 4))
plt.scatter(df['sepal_length'], df['sepal_width'], c=df['cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

print("Level 2-Task 3 completed.")
