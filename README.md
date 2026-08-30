# 🌾 PaddyYield Intelligence — Advanced Streamlit System

A four-model machine-learning dashboard based on the supplied `ABCCC.ipynb` and `paddydataset.csv`.

## Four modelling tracks
1. Logistic Regression
2. ANN (MLPClassifier)
3. Random Forest
4. KNN

## Advanced features
- Executive analytics dashboard
- Four-model leaderboard
- Single-model prediction
- Compare all four predictions
- Soft-voting probability ensemble
- What-if scenario simulator
- Confusion matrix + multiclass ROC curves
- 5-fold cross-validation summaries
- Learning curves
- Permutation feature importance
- Interactive EDA / data explorer
- Session prediction history + CSV export
- GitHub + Streamlit Cloud deployment ready

## Target
- Low: 0–10,000 kg
- Moderate: 10,001–20,000 kg
- High: 20,001–30,000 kg
- Very High: 30,001–40,000 kg

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
Push all files to GitHub and select `app.py` as the main file.

## Important methodology note
The model bundle follows the supplied notebook's modelling setup and uses a 75/25 stratified train-test split with random_state=42. The bundled models use tuned/selected configurations consistent with the notebook's modelling sections. The application is for academic/analytical demonstration and is not a guarantee of actual yield.
