# Census Income Predictor

A Streamlit app that predicts whether a person's income exceeds **50K USD** from Census
data, and lets you compare several classification models.

## Pages

- **Home** – project overview.
- **Data exploration** – descriptive statistics and charts on `census.csv`.
- **Model training** – trains and compares KNN, Decision Tree, Random Forest, Gradient
  Boosting, Logistic Regression and SVM, with a configurable test size and random state;
  shows metrics, the best model and its confusion matrix.
- **Prediction** – form for age, education level, capital gain/loss and hours per week.

The target is binary: `<=50K` (class 0) versus `>50K` (class 1). Features are the numeric
columns of the dataset, standardised with `StandardScaler`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run census_app.py
```

## Files

| File | Purpose |
|---|---|
| `census_app.py` | Streamlit application (exploration, training, prediction) |
| `census.csv` | Training dataset |
| `census.pkl`, `best_model.pkl` | Saved models |
| `requirements.txt` | Dependencies |

## Stack

Python, Streamlit, pandas, NumPy, scikit-learn, joblib, matplotlib.

## Known limitations

- The Prediction page loads `census.pkl`, `scaler.pkl` and `feature_names.pkl`; the last two
  are not in this repository, so the page shows "no model found" until they are provided.
- Only numeric features are used; categorical variables are ignored.
