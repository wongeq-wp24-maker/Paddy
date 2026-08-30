import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "paddydataset.csv"
MODEL_PATH = BASE_DIR / "paddy_yield_model.joblib"

TARGET = "Paddy yield(in Kg)"
EXCLUDED = [
    "Hectares", "Seedrate(in Kg)", "LP_Mainfield(in Tonnes)",
    "Nursery area (Cents)", "LP_nurseryarea(in Tonnes)", "DAP_20days",
    "Weed28D_thiobencarb", "Urea_40Days", "Potassh_50Days",
    "Micronutrients_70Days", "Pest_60Day(in ml)"
]

# Same cleaning and target construction as the notebook.
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df = df.drop_duplicates().reset_index(drop=True)

bins = [0, 10000, 20000, 30000, 40000]
labels = ["Low", "Moderate", "High", "Very High"]
df["Paddy Yield Category"] = pd.cut(
    df[TARGET], bins=bins, labels=labels, right=False
)

features = [c for c in df.columns if c not in EXCLUDED and c not in [TARGET, "Paddy Yield Category"]]
X = df[features].copy()
y = df["Paddy Yield Category"].copy()

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["lbfgs", "liblinear"],
    "classifier__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
)
search.fit(X_train, y_train)

joblib.dump(search.best_estimator_, MODEL_PATH)
print("Best parameters:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))
print("Saved:", MODEL_PATH)
