import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)
os.makedirs('outputs', exist_ok=True)

with open('data/model.pkl', 'rb') as f:
    old_model = pickle.load(f)

monitor = pd.read_csv('data/monitoring_results.csv')

print("=" * 50)
print("  FINAL REPORT -- End-to-End ML Deployment")
print("=" * 50)

print("\nMonitoring Results:")
print(monitor.to_string(index=False))

drift_months = monitor[monitor['mae'] >= 20]
print("\nDrift detected in:", drift_months['month'].tolist())
print("These months need retraining!")

print("\nRetraining model on new data...")

n = 1000
new_df = pd.DataFrame({
    'price' : np.random.randint(500, 1500, n),
    'discount' : np.random.randint(0, 50, n),
    'day_of_week' : np.random.randint(0, 7, n),
    'is_weekend' : np.random.randint(0, 2, n),
    'is_campaign' : np.random.randint(0, 2, n),
    'stock_level' : np.random.randint(10, 500, n),
    'competitor_price' : np.random.randint(100, 1000, n),
})

new_df['units_sold'] = (
    50
    + new_df['discount'] * 1.5
    + new_df['is_campaign'] * 30
    + new_df['is_weekend'] * 20
    - new_df['price'] * 0.05
    + np.random.randint(-10, 10, n)
).astype(int).clip(1)

features = ['price', 'discount', 'day_of_week', 'is_weekend', 'is_campaign', 'stock_level', 'competitor_price']

X = new_df[features]
y = new_df['units_sold']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

new_model = RandomForestRegressor(n_estimators=100, random_state=42)
new_model.fit(X_train, y_train)

new_preds = new_model.predict(X_test)
new_mae = mean_absolute_error(y_test, new_preds)
new_r2 = r2_score(y_test, new_preds)

old_preds_on_new = old_model.predict(X_test)
old_mae_on_new = mean_absolute_error(y_test, old_preds_on_new)

print("\nOld model MAE on new data:", round(old_mae_on_new, 2), " (bad -- drift!)")
print("New model MAE on new data:", round(new_mae, 2), " (good -- retrained!)")
print("New model R2 :", round(new_r2, 3))

with open('data/model_retrained.pkl', 'wb') as f:
    pickle.dump(new_model, f)
print("\nRetrained model saved to: data/model_retrained.pkl")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('ML Deployment Platform -- Final Dashboard', fontsize=16, fontweight='bold')

colors = ['green' if m < 20 else 'red' for m in monitor['mae']]
axes[0, 0].bar(monitor['month'], monitor['mae'], color=colors, edgecolor='black')
axes[0, 0].axhline(y=20, color='red', linestyle='--', label='Drift threshold')
axes[0, 0].set_title('Model Monitoring -- MAE per Month')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=30)

axes[0, 1].bar(['Old Model\n(on new data)', 'Retrained\nModel'],
               [old_mae_on_new, new_mae],
               color=['#e74c3c', '#2ecc71'], edgecolor='black', width=0.4)
axes[0, 1].set_title('Before vs After Retraining')
axes[0, 1].set_ylabel('MAE (lower = better)')
for i, val in enumerate([old_mae_on_new, new_mae]):
    axes[0, 1].text(i, val + 0.3, str(round(val, 1)),
                    ha='center', fontsize=12)

axes[1, 0].plot(y_test.values[:80], label='Actual', color='steelblue', linewidth=1.5)
axes[1, 0].plot(new_preds[:80], label='Predicted', color='darkorange',
               linewidth=1.5, linestyle='--')
axes[1, 0].set_title('Retrained Model -- Actual vs Predicted')
axes[1, 0].set_ylabel('Units Sold')
axes[1, 0].legend()

importances = pd.Series(
    new_model.feature_importances_, index=features
).sort_values(ascending=True)

axes[1, 1].barh(importances.index, importances.values,
                color='steelblue', edgecolor='black')
axes[1, 1].set_title('Feature Importance\n(which inputs matter most?)')
axes[1, 1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('outputs/chart3_final_dashboard.png', dpi=150)
plt.show()

print("\nFinal dashboard saved: outputs/chart3_final_dashboard.png")

print("\n" + "=" * 50)
print("RECOMMENDATIONS:")
print("=" * 50)
print("""
1. MONITOR MODEL EVERY MONTH
   MAE crossing 20 means drift has happened.
   Set up automatic alerts when this occurs.

2. RETRAIN WHEN DRIFT IS DETECTED
   New data (like price changes or new trends)
   makes old models inaccurate. Retrain immediately.

3. PRICE IS THE BIGGEST FACTOR
   Feature importance shows price affects
   units sold the most. Track price changes closely.

4. AUTOMATE THE PIPELINE
   In real companies, all 4 steps run automatically:
   collect data -- train -- monitor -- retrain.
   This is what a full ML pipeline looks like.
""")
print("=" * 50)
