import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

app = Flask(__name__)

# -------------------------
# 1. Train model if not already saved
# -------------------------
if not (os.path.exists("preprocessor.joblib") and os.path.exists("xgb_model.joblib")):
    print("⚡ Training model and creating joblib files...")

    np.random.seed(42)
    n = 500
    sqft = np.random.randint(500, 4000, n)
    rooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 4, n)
    age = np.random.randint(0, 50, n)
    location = np.random.choice(["Downtown", "Suburb", "Countryside"], n)

    price = (
        sqft * np.random.uniform(200, 300) +
        rooms * 50000 +
        bathrooms * 30000 -
        age * 1000 +
        np.where(location == "Downtown", 150000,
                 np.where(location == "Suburb", 80000, 30000)) +
        np.random.normal(0, 50000, n)
    )

    df = pd.DataFrame({
        "sqft": sqft,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "age": age,
        "location": location,
        "price": price.astype(int)
    })
    df.to_csv("housing.csv", index=False)

    X = df.drop("price", axis=1)
    y = df["price"]

    numeric_features = ["sqft", "rooms", "bathrooms", "age"]
    categorical_features = ["location"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(model, "xgb_model.joblib")

    print("✅ Training complete & files saved!")

# -------------------------
# 2. Load model & preprocessor
# -------------------------
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("xgb_model.joblib")

# -------------------------
# 3. Flask routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sqft = int(request.form["sqft"])
        rooms = int(request.form["rooms"])
        bathrooms = int(request.form["bathrooms"])
        age = int(request.form["age"])
        location = request.form["location"]

        input_df = pd.DataFrame([{
            "sqft": sqft,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "age": age,
            "location": location
        }])

        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]

        return render_template("form.html",
                               prediction_text=f"🏠 Estimated Price: ₹{prediction:,.0f}")
    except Exception as e:
        return render_template("form.html",
                               prediction_text=f"⚠️ Error: {str(e)}")

# -------------------------
# 4. Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
