import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

# Load dataset
def load_data(path=r'C:\Users\admin\Desktop\creditscoreproject\data\credit_risk_1_lakh_dataset.csv'):
    df = pd.read_csv(path)
    return df

# Preprocess dataset
def preprocess_data(df, target_column='Credit_Risk'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Label encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Encode target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y

# Train model and save
def train_model(X, y, save_path='models/model.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model, X_train, X_test, y_train, y_test

# Load trained model
def load_model(path='models/model.pkl'):
    model = joblib.load(path)
    return model

# Predict new data
def predict(model, input_df):
    return model.predict(input_df), model.predict_proba(input_df)

# Explain predictions using SHAP
def explain(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, explainer

