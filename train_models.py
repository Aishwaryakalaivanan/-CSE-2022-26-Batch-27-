import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# ===== Load your dataset =====
df = pd.read_csv(r"C:\Users\admin\Desktop\creditscoreproject\data\credit_risk_1_lakh_dataset.csv")  # replace with your file

# ===== Features and target =====
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# ===== Identify categorical columns =====
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ===== Preprocessor for categorical features =====
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False)
, categorical_cols)
    ],
    remainder='passthrough'
)

# ===== Transform features =====
X_processed = preprocessor.fit_transform(X)

# ===== Encode target =====
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===== Split data =====
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)

# ===== Train XGBoost model =====
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# ===== Save model and preprocessor =====
joblib.dump(model, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model and preprocessor saved successfully!")

