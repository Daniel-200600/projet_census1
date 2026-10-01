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
| `census.pkl`, `scaler.pkl`, `feature_names.pkl` | Saved model, scaler and feature names |
| `requirements.txt` | Dependencies |

## Stack

Python, Streamlit, pandas, NumPy, scikit-learn, joblib, matplotlib.

## Model files

The **Model training** page saves the best model (`census.pkl`), the fitted
`StandardScaler` (`scaler.pkl`) and the feature names (`feature_names.pkl`), which the
**Prediction** page reloads. If these files are missing, or were created with an incompatible
scikit-learn version, the Prediction page automatically trains a default Random Forest on
`census.csv` instead, so it always works. The committed files were generated with
scikit-learn 1.9.1.

## Known limitations

- Only numeric features are used (age, education level, capital gain/loss, hours per week);
  categorical variables are ignored.
- Training all six models, including the SVM, takes a few minutes.
