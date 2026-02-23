# ==============================
# Streamlit Credit Risk Prediction App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ======================================
# 🔢 Convert Probability to Credit Score
# ======================================

def probability_to_score(prob):
    # Convert probability (0–1) to credit score (300–850)
    return 300 + prob * 550

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Prediction App")
st.write("Machine Learning + Explainable AI (SHAP)")

# ==============================
# 1️⃣ LOAD MODEL & FEATURES
# ==============================

model = joblib.load(r"C:\Users\admin\Desktop\creditscoreproject\models\model.pkl")
feature_columns = joblib.load(r"C:\Users\admin\Desktop\creditscoreproject\models\feature_columns.pkl")

# ==============================
# 2️⃣ INPUT SECTION
# ==============================

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(input_df.head())

else:
    st.subheader("Or Enter Details Manually")

    input_dict = {
        "age": st.number_input("Age", 18, 100, 30),
        "income": st.number_input("Income", 0, 1000000, 50000),
        "sex": st.selectbox("Sex", ["male", "female"]),
        "employment_status": st.selectbox("Employment Status", ["employed", "unemployed", "self-employed"]),
        "credit_history": st.selectbox("Credit History", ["good", "bad"]),
        "repayment_behavior": st.selectbox("Repayment Behavior", ["on_time", "late"])
    }

    input_df = pd.DataFrame([input_dict])

st.subheader("Input Data")
st.dataframe(input_df)

# ==============================
# 3️⃣ ENCODE CATEGORICAL DATA
# ==============================

for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = LabelEncoder().fit_transform(input_df[col])

# ==============================
# 4️⃣ ALIGN FEATURES WITH TRAINING
# ==============================
# If feature_columns is dict, convert to list
if isinstance(feature_columns, dict):
    feature_list = list(feature_columns.keys())
else:
    feature_list = list(feature_columns)
# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep only training columns in correct order
input_df = input_df.reindex(columns=feature_columns, fill_value=0)


st.write("Final Input Shape:", input_df.shape)

# ==============================
# 5️⃣ PREDICTION
# ==============================

prediction = model.predict(input_df)
st.subheader("Prediction Result")
st.write(prediction)

# ======================================
# 📊 Clean Structured Waterfall Style
# ======================================

st.markdown("## 🔎 Single Prediction Explanation")

import numpy as np

# ---- Prediction probability ----
proba = model.predict_proba(input_df)[0][1]

# ---- Convert to score scale ----
base_probability = 0.5  # neutral baseline
base_score = probability_to_score(base_probability)
final_score = probability_to_score(proba)

# ---- SHAP Calculation ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

if isinstance(shap_values, list):
    shap_vals = shap_values[1][0]
    expected_value = explainer.expected_value[1]
else:
    shap_vals = shap_values[0]
    expected_value = explainer.expected_value
# Convert SHAP log-odds → score impact
score_impacts = shap_vals * 550

# Create dataframe
impact_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Impact": score_impacts
})

impact_df["abs"] = impact_df["Impact"].abs()
impact_df = impact_df.sort_values(by="abs", ascending=False).drop(columns="abs")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==========================
# CREATE WATERFALL FIGURE
# ==========================
fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor("#f0f0f0")
ax.axis("off")

ax.text(0.02, 0.95, "SHAP Waterfall Plot",
        fontsize=18, fontweight="bold", color="#1f3c88")

ax.text(0.02, 0.90, "Single Prediction Explanation",
        fontsize=14, fontweight="bold", color="#1f3c88")

# Base value
ax.text(0.02, 0.83, f"Base Value: {base_score:.1f}",
        fontsize=14)

# Base box
ax.add_patch(Rectangle((0.05, 0.70), 0.15, 0.08, color="gray"))
ax.text(0.125, 0.74, f"{base_score:.0f}",
        ha="center", va="center",
        color="white", fontsize=14, fontweight="bold")

# Take top 3 features
top3 = impact_df.head(3)

x_start = 0.30
y_start = 0.68

running_score = base_score

for i, row in enumerate(top3.itertuples()):

    impact = int(row.Impact)
    running_score += impact

    color = "#2ca02c" if impact > 0 else "#d62728"
    sign = "+" if impact > 0 else ""

    # Diagonal waterfall blocks
    ax.add_patch(Rectangle(
        (x_start + i*0.20, y_start - i*0.08),
        0.15, 0.08,
        color=color
    ))

    ax.text(
        x_start + i*0.20 + 0.075,
        y_start - i*0.08 + 0.04,
        f"{sign}{impact}",
        ha="center", va="center",
        fontsize=14,
        color="white",
        fontweight="bold"
    )

    # Feature text on left
    ax.text(
        0.05,
        0.55 - i*0.07,
        f"➤ {row.Feature} {sign}{impact}",
        fontsize=14
    )

# Final box
credit_label = "Good Credit" if final_score >= 600 else "Bad Credit"

ax.add_patch(Rectangle((0.75, 0.30), 0.20, 0.20, color="#1f77b4"))

ax.text(0.85, 0.40,
        f"{int(final_score)}\n{credit_label}",
        ha="center", va="center",
        fontsize=18,
        color="white",
        fontweight="bold")

# Final prediction line
total_impact = int(top3["Impact"].sum())

ax.text(0.05, 0.20,
        f"+{total_impact}  Final Prediction: {int(final_score)} ({credit_label})",
        fontsize=16,
        fontweight="bold",
        color="#1f3c88")

st.pyplot(fig)


# ======================================
# 📊 SHAP Waterfall Plot
# ======================================

st.markdown("## 📊 Detailed Feature Impact (Waterfall)")

import matplotlib.pyplot as plt

# Create SHAP explanation object
shap_explanation = shap.Explanation(
    values=shap_vals,
    base_values=expected_value,
    data=input_df.iloc[0],
    feature_names=input_df.columns
)

# Plot waterfall
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_explanation, show=False)

st.pyplot(fig)
plt.close(fig)