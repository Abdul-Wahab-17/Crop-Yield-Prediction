import pandas as pd

df = pd.read_csv('/home/roger/Downloads/eee/data/processed/master_yield_dataset.csv')

print(f"Row count: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nMissing values:")
print(df.isnull().sum())

print("\nSample row:")
print(df.iloc[0])
