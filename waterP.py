import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import zipfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import logging

# Set Streamlit page config
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")

# Enable logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# ---------------------- LOAD DATASET ----------------------

@st.cache_data
def load_data():
    try:
        with zipfile.ZipFile(r"D:\dataset\archive.zip") as z:
            with z.open("water_potability.csv") as f:
                df = pd.read_csv(f)
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

# ---------------------- DATA PREPROCESSING ----------------------

def preprocess(df):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, imputer, scaler, X.columns

X, y, imputer, scaler, feature_names = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------- MODEL TRAINING ----------------------

@st.cache_resource
def train_model():
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

model = train_model()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# ---------------------- USER INPUT ----------------------

def sidebar_input():
    with st.sidebar:
        st.header("🔧 Enter Water Quality Parameters")
        return {
            "ph": st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1),
            "Hardness": st.number_input("Hardness (mg/L)", 0.0, 500.0, 180.0),
            "Solids": st.number_input("Total Solids (ppm)", 0.0, 50000.0, 10000.0),
            "Chloramines": st.number_input("Chloramines (ppm)", 0.0, 20.0, 7.0, 0.1),
            "Sulfate": st.number_input("Sulfate (mg/L)", 0.0, 600.0, 250.0),
            "Conductivity": st.number_input("Conductivity (μS/cm)", 0.0, 2000.0, 400.0),
            "Organic_carbon": st.number_input("Organic Carbon (ppm)", 0.0, 30.0, 10.0),
            "Trihalomethanes": st.number_input("Trihalomethanes (μg/L)", 0.0, 200.0, 70.0),
            "Turbidity": st.number_input("Turbidity (NTU)", 0.0, 10.0, 3.0),
        }

def validate(inputs):
    warnings = []
    ideal = {
        "pH": (6.5, 8.5),
        "Hardness": (0, 200),
        "Solids": (0, 500),
        "Chloramines": (0, 4),
        "Sulfate": (0, 250),
        "Conductivity": (0, 500),
        "Organic_carbon": (0, 2),
        "Trihalomethanes": (0, 80),
        "Turbidity": (0, 1),
    }

    for i, (key, val) in enumerate(inputs.items()):
        if key in ideal:
            min_val, max_val = ideal[key]
            if not (min_val <= val <= max_val):
                warnings.append(f"{key} is outside ideal range ({min_val}-{max_val})")

    return warnings

def predict(input_data):
    try:
        X_input = np.array([list(input_data.values())])
        X_proc = scaler.transform(imputer.transform(X_input))
        pred = model.predict(X_proc)[0]
        conf = model.predict_proba(X_proc)[0][1]
        return pred, conf
    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        st.error("Error while making prediction.")
        return None, None

# ---------------------- APP LAYOUT ----------------------

st.title("💧 Smart Water Potability Predictor")
st.markdown("Check whether your water sample is potable based on scientific parameters.")

params = sidebar_input()

if st.sidebar.button("🔍 Check Potability"):
    warnings = validate(params)
    if warnings:
        st.warning("⚠️ " + "\n⚠️ ".join(warnings))

    prediction, confidence = predict(params)
    if prediction is not None:
        st.subheader("🧪 Prediction Result:")
        if prediction == 1:
            st.success(f"✅ Potable (Confidence: {confidence:.2%})")
        else:
            st.error(f"❌ Not Potable (Confidence: {confidence:.2%})")

        with st.expander("🔍 SHAP Explanation"):
            explainer = shap.TreeExplainer(model)
            X_input = np.array([list(params.values())])
            X_proc = scaler.transform(imputer.transform(X_input))
            shap_vals = explainer.shap_values(X_proc)

            fig, ax = plt.subplots()
            shap.summary_plot(shap_vals, features=X_proc, feature_names=feature_names, show=False)
            st.pyplot(fig)

# ---------------------- PERFORMANCE ----------------------

with st.expander("📈 Model Performance Metrics"):
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("F1 Score (Potable)", f"{report['1']['f1-score']:.2%}")
    st.dataframe(pd.DataFrame(report).transpose())
