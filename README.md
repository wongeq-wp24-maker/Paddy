# Paddy Yield Category Prediction — Streamlit

This project deploys the paddy-yield classification work from `ABCCC.ipynb` as a Streamlit web application.

## Model

The notebook creates four target categories from `Paddy yield(in Kg)`:

- **Low:** 0–9,999 kg
- **Moderate:** 10,000–19,999 kg
- **High:** 20,000–29,999 kg
- **Very High:** 30,000–39,999 kg

The final comparison in the notebook reports Logistic Regression as the strongest model, with **97.44% accuracy** and **97.78% macro F1-score**.

## Files

- `app.py` — Streamlit application
- `paddydataset.csv` — dataset used by the notebook
- `paddy_yield_model.joblib` — serialized tuned Logistic Regression pipeline
- `train_model.py` — script to rebuild the model
- `ABCCC.ipynb` — original analysis notebook
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in this folder.
3. Make sure `app.py`, `paddydataset.csv`, and `paddy_yield_model.joblib` are in the repository root.
4. In Streamlit Community Cloud, select the GitHub repository and set the main file to `app.py`.
5. Deploy.

The app uses the serialized pipeline, so the same scaling and one-hot encoding are applied automatically to the input values.
