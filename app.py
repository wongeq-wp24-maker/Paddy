import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Paddy Yield Category Predictor", page_icon="🌾", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "paddy_yield_model.joblib"

TARGET = "Paddy yield(in Kg)"
EXCLUDED = [
    "Hectares", "Seedrate(in Kg)", "LP_Mainfield(in Tonnes)",
    "Nursery area (Cents)", "LP_nurseryarea(in Tonnes)", "DAP_20days",
    "Weed28D_thiobencarb", "Urea_40Days", "Potassh_50Days",
    "Micronutrients_70Days", "Pest_60Day(in ml)"
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    df = pd.read_csv(BASE_DIR / "paddydataset.csv")
    df.columns = df.columns.str.strip()
    return df

st.title("🌾 Paddy Yield Category Predictor")
st.write("Predict the expected paddy-yield category using the tuned Logistic Regression model from the analysis notebook.")

try:
    model = load_model()
    df = load_dataset()
except Exception as e:
    st.error(f"Unable to load the application files: {e}")
    st.stop()

selected_features = [c for c in df.columns if c not in EXCLUDED and c != TARGET]

st.info("The model uses the same feature-selection approach as the notebook: the 11 highly redundant plot-size-related variables are excluded.")

with st.form("prediction_form"):
    st.subheader("Farm and field information")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        agriblock = st.selectbox("Agriblock", sorted(df["Agriblock"].dropna().unique()))
        variety = st.selectbox("Variety", sorted(df["Variety"].dropna().unique()))
    with c2:
        soil = st.selectbox("Soil Types", sorted(df["Soil Types"].dropna().unique()))
        nursery = st.selectbox("Nursery", sorted(df["Nursery"].dropna().unique()))
    with c3:
        wind_d1 = st.selectbox("Wind Direction D1-D30", sorted(df["Wind Direction_D1_D30"].dropna().unique()))
        wind_d2 = st.selectbox("Wind Direction D31-D60", sorted(df["Wind Direction_D31_D60"].dropna().unique()))
    with c4:
        wind_d3 = st.selectbox("Wind Direction D61-D90", sorted(df["Wind Direction_D61_D90"].dropna().unique()))
        wind_d4 = st.selectbox("Wind Direction D91-D120", sorted(df["Wind Direction_D91_D120"].dropna().unique()))

    st.subheader("Weather and field variables")
    numeric_features = [c for c in selected_features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in selected_features if c not in numeric_features]

    defaults = {}
    for col in numeric_features:
        defaults[col] = float(df[col].median())

    cols = st.columns(3)
    numeric_values = {}
    for i, col in enumerate(numeric_features):
        with cols[i % 3]:
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            default_v = defaults[col]
            if min_v == max_v:
                numeric_values[col] = default_v
                st.number_input(col, value=default_v, disabled=True)
            else:
                numeric_values[col] = st.number_input(col, min_value=min_v, max_value=max_v, value=default_v, format="%.3f")

    submitted = st.form_submit_button("Predict Yield Category", type="primary")

if submitted:
    input_data = {
        "Agriblock": agriblock,
        "Variety": variety,
        "Soil Types": soil,
        "Nursery": nursery,
        "Wind Direction_D1_D30": wind_d1,
        "Wind Direction_D31_D60": wind_d2,
        "Wind Direction_D61_D90": wind_d3,
        "Wind Direction_D91_D120": wind_d4,
    }
    input_data.update(numeric_values)

    input_df = pd.DataFrame([input_data], columns=selected_features)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    confidence = float(probabilities.max())

    st.subheader("Prediction")
    st.success(f"Predicted Paddy Yield Category: **{prediction}**")
    st.metric("Prediction confidence", f"{confidence:.2%}")

    probability_df = pd.DataFrame({
        "Category": classes,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)
    probability_df["Probability"] = probability_df["Probability"].map(lambda x: f"{x:.2%}")

    st.subheader("Class probabilities")
    st.dataframe(probability_df, use_container_width=True, hide_index=True)

    ranges = {
        "Low": "0–9,999 kg",
        "Moderate": "10,000–19,999 kg",
        "High": "20,000–29,999 kg",
        "Very High": "30,000–39,999 kg",
    }
    st.caption(f"Category range: {ranges.get(prediction, 'See notebook target definition.')}")

with st.expander("About this model"):
    st.write("Target categories follow the notebook: Low, Moderate, High and Very High, created with 10,000 kg category intervals.")
    st.write("The notebook reports the Logistic Regression model as the strongest model in the final comparison: 97.44% accuracy and 97.78% macro F1-score.")
    st.write("This Streamlit app loads a serialized tuned Logistic Regression pipeline so the preprocessing is applied consistently to user input.")
