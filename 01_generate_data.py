import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'price' : np.random.randint(100, 1000, n),
    'discount' : np.random.randint(0, 50, n),
    'day_of_week' : np.random.randint(0, 7, n),
    'is_weekend' : np.random.randint(0, 2, n),
    'is_campaign' : np.random.randint(0, 2, n),
    'stock_level' : np.random.randint(10, 500, n),
    'competitor_price' : np.random.randint(100, 1000, n),
})

df['units_sold'] = (
    50
    + df['discount'] * 1.5
    + df['is_campaign'] * 30
    + df['is_weekend'] * 20
    - df['price'] * 0.05
    + np.random.randint(-10, 10, n)
).astype(int).clip(1)

os.makedirs('data', exist_ok=True)
df.to_csv('data/sales_data.csv', index=False)

print("Dataset created! Rows:", len(df))
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nAverage units sold per day:", round(df['units_sold'].mean(), 1))