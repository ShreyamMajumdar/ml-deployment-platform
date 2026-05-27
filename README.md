# ⚙️ End-to-End ML Deployment Platform
 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![MLOps](https://img.shields.io/badge/MLOps-Drift%20Detection-red)
 
## 📌 Overview
A complete MLOps pipeline that simulates the full machine learning
lifecycle -- from data generation and model training to deployment,
monthly performance monitoring, drift detection and automatic retraining.
 
## 🎯 Objective
- Train and deploy a Random Forest regression model
- Monitor model performance every month after deployment
- Detect when model accuracy drops due to data drift
- Automatically retrain the model on new data when drift is detected
 
## 📂 Project Structure
```
ml_deployment/
│
├── data/
│   ├── sales_data.csv
│   ├── model.pkl
│   ├── model_retrained.pkl
│   └── monitoring_results.csv
│
├── outputs/
│   └── (3 charts saved here)
│
├── 01_generate_data.py
├── 02_train_model.py
├── 03_monitor.py
└── 04_retrain_and_insights.py
```
 
## 📊 Dataset
- **Type:** Synthetic daily sales data
- **Size:** 1000 rows
- **Features:** price, discount, day_of_week, is_weekend,
  is_campaign, stock_level, competitor_price
- **Target:** units_sold per day
 
## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| pandas | Data handling |
| numpy | Data generation |
| matplotlib | Monitoring charts |
| scikit-learn | Random Forest and metrics |
| pickle | Save and load models |
 
## 🔄 ML Pipeline
```
Generate Data
     ↓
Train Random Forest Model
     ↓
Deploy Model (save with pickle)
     ↓
Monitor MAE Every Month
     ↓
MAE > 20 means DRIFT DETECTED
     ↓
Retrain on New Data
     ↓
Redeploy Improved Model
```
 
## 📉 Drift Simulation Results
| Month | Price Range | MAE | Status |
|-------|-------------|-----|--------|
| Month 1 | Rs. 100-1000 | Low | OK |
| Month 2 | Rs. 100-1000 | Low | OK |
| Month 3 | Rs. 100-1000 | Low | OK |
| Month 4 | Rs. 700-1500 | High | DRIFT DETECTED |
| Month 5 | Rs. 700-1500 | High | DRIFT DETECTED |
| Month 6 | Rs. 700-1500 | High | DRIFT DETECTED |
 
## 📈 Key Findings
- Model performed well for first 3 months with MAE below threshold
- Drift detected from month 4 when price distribution shifted
- Old model MAE on new data was 3-5x higher than original
- Retrained model restored accuracy to near-original levels
- Price was the most important feature in both model versions
 
## 🚀 How to Run
```bash
pip install pandas numpy matplotlib scikit-learn
 
python 01_generate_data.py
python 02_train_model.py
python 03_monitor.py
python 04_retrain_and_insights.py
```
