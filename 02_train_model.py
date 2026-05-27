import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('data/sales_data.csv')

print("Data loaded! Rows:", len(df))

features = ['price', 'discount', 'day_of_week', 'is_weekend', 'is_campaign', 'stock_level', 'competitor_price']
target   = 'units_sold'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

print("\nTraining model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nModel Results:")
print(" MAE (avg error in units) :", round(mae, 2))
print(" R2  (how well it fits) :", round(r2, 3), " (1.0 = perfect)")

os.makedirs('data', exist_ok=True)
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to: data/model.pkl")

results = pd.DataFrame({
    'actual' : y_test.values,
    'predicted' : preds.round(1)
})
results.to_csv('data/test_results.csv', index=False)
print("Test results saved to: data/test_results.csv")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:80],  label='Actual',    color='steelblue', linewidth=1.5)
plt.plot(preds[:80], label='Predicted', color='darkorange', linewidth=1.5, linestyle='--')
plt.title('Actual vs Predicted -- Units Sold', fontsize=14)
plt.xlabel('Sample')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/chart1_actual_vs_predicted.png')
plt.show()

print("Chart saved.")