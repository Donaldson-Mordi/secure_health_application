# train_model.py
# I'm training a an AI model (Logistic Regression) to predict stroke risk
# using the SAME MongoDB patients data that my app already uses.

from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# --- 1. I'm connecting to the same MongoDB as my Flask app ---
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["secure_health_db"]
patients_collection = mongo_db["patients"]

# --- 2. I'm pulling out the fields I want to use as features ---
# I'll use: age, hypertension, heart_disease, avg_glucose_level, bmi
# and the target label: stroke (0/1)
X = []
y = []

for doc in patients_collection.find({}, {
    "_id": 0,
    "age": 1,
    "hypertension": 1,
    "heart_disease": 1,
    "avg_glucose_level": 1,
    "bmi": 1,
    "stroke": 1,
}):
    try:
        age = float(doc.get("age"))
        htn = int(doc.get("hypertension"))
        hd = int(doc.get("heart_disease"))
        glu = float(doc.get("avg_glucose_level"))
        bmi = doc.get("bmi")

        # I'm skipping rows with missing BMI
        if bmi in (None, "", "N/A"):
            continue
        bmi = float(bmi)

        stroke = int(doc.get("stroke"))
    except (TypeError, ValueError):
        # If anything is wrong, I skip this record
        continue

    X.append([age, htn, hd, glu, bmi])
    y.append(stroke)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} records for training.")

if len(X) < 50:
    print("⚠ Not enough data to train a reliable model.")
else:
    # --- 3. I'm splitting into train and test sets so I can check accuracy ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. I'm training a simple Logistic Regression model ---
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)

    # --- 5. I'm checking basic accuracy on the test set ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test accuracy: {acc:.3f}")

    # --- 6. I'm saving the trained model to a .pkl file so Flask can load it ---
    joblib.dump(model, "stroke_model.pkl")
    print("✅ Saved model to stroke_model.pkl")