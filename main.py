import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = r"C:\Mobile Phone Pricing\dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.info())

# -----------------------------
# EDA
# -----------------------------
print("\nPrice range distribution:")
print(df['price_range'].value_counts())

# -----------------------------
# PREPARE DATA
# -----------------------------
X = df.drop(columns=['price_range'])
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = rf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
package = {'model': rf, 'scaler': scaler, 'features': X.columns.tolist()}
joblib.dump(package, os.path.join(MODEL_DIR, 'rf_price_model.joblib'))
print("✅ Model saved to models/rf_price_model.joblib")

