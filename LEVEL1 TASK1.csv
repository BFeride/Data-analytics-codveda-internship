import pandas as pd

# 1. Read dataset and clean missing/duplicate values
df = pd.read_csv('1.iris.csv').dropna().drop_duplicates()

# 2. Save cleaned file
df.to_csv('cleaned_iris.csv',index=False)
print("Level 1 - Task 1 completed.")
