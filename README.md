# 🌾 AgriTech AI – Smart Farming Advisory System

## 📌 Overview
AgriTech AI is a universal, offline-capable machine learning system designed to assist farmers with data-driven crop and yield insights. The system automatically analyzes agricultural datasets, predicts crop yield or classifies crop outcomes, and converts model predictions into simple, actionable advisory messages for farmers.

The solution bridges the gap between advanced AI models and real-world farming needs by providing explainable results in multiple local languages.

---

## 🎯 Key Features
- Automatic detection of **Regression** (yield prediction) or **Classification** (crop/risk category)
- Dynamic model selection based on dataset size
- Supports multiple models:
  - RandomForest
  - HistGradientBoosting
- Robust preprocessing for numeric and categorical data
- Farmer-friendly advisory output in:
  - English
  - Hindi
  - Kannada
- Fully **offline**, CLI-based execution
- Reproducible results with fixed random seed

---

## 🧠 Models Used
| Task Type | Small Dataset | Large Dataset |
|----------|---------------|---------------|
| Regression | RandomForestRegressor | HistGradientBoostingRegressor |
| Classification | RandomForestClassifier | HistGradientBoostingClassifier |

---

## 🌾 Farmer Advisory System
Model predictions are converted into three intuitive categories:
- **LOW** – Conditions are unfavorable, improvement needed
- **MEDIUM** – Acceptable conditions, scope for improvement
- **HIGH** – Favorable conditions, continue best practices

Each category is explained with clear farming actions such as irrigation control, fertilizer usage, pest monitoring, and crop selection. Farmers can choose their preferred language after prediction.

---

## ▶️ How to Run

### Required Command
```bash
python main.py --train data/train.csv --test data/test.csv --out output/predictions.csv
