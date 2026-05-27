import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error
import os

np.random.seed(99)
os.makedirs('outputs', exist_ok=True)

with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded.")
print("\nSimulating 6 months of live predictions...")
print("(Checking if model performance drops over time)")

features = ['price', 'discount', 'day_of_week', 'is_weekend', 'is_campaign', 'stock_level', 'competitor_price']
months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
mae_scores = []

print("\nMAE per month (lower = better):")
print("-" * 35)

for i, month in enumerate(months):
    n = 100

    if i >= 3:
        price = np.random.randint(700, 1500, n)
    else:
        price = np.random.randint(100, 1000, n)

    new_data = pd.DataFrame({
        'price' : price,
        'discount' : np.random.randint(0, 50, n),
        'day_of_week' : np.random.randint(0, 7, n),
        'is_weekend' : np.random.randint(0, 2, n),
        'is_campaign' : np.random.randint(0, 2, n),
        'stock_level' : np.random.randint(10, 500, n),
        'competitor_price' : np.random.randint(100, 1000, n),
    })

    actual = (
        50
        + new_data['discount'] * 1.5
        + new_data['is_campaign'] * 30
        + new_data['is_weekend'] * 20
        - new_data['price'] * 0.05
        + np.random.randint(-10, 10, n)
    ).astype(int).clip(1)

    predicted = model.predict(new_data[features])
    mae = mean_absolute_error(actual, predicted)
    mae_scores.append(round(mae, 2))

    status = "OK" if mae < 20 else "WARNING -- drift detected!"
    print(" {month}: MAE = {round(mae, 2):>6}  |  {status}")

plt.figure(figsize=(10, 5))
colors = ['green' if m < 20 else 'red' for m in mae_scores]
bars = plt.bar(months, mae_scores, color=colors, edgecolor='black')

for bar, val in zip(bars, mae_scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             str(val), ha='center', fontsize=11)

plt.axhline(y=20, color='red', linestyle='--', label='Drift threshold (MAE = 20)')
plt.title('Model Performance Over 6 Months\n(red bars = drift detected, retrain needed)', fontsize=13)
plt.ylabel('MAE -- Average Error in Units')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/chart2_model_monitoring.png')
plt.show()

print("\nChart saved: outputs/chart2_model_monitoring.png")

pd.DataFrame({'month': months, 'mae': mae_scores}).to_csv(
    'data/monitoring_results.csv', index=False
)
