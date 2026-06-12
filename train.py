import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import pickle

# ── 1. მონაცემების ჩატვირთვა ──────────────────────────────────────────────────
df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
# social_interaction_level: ტექსტი → რიცხვი
social_map = {"low": 0, "medium": 1, "high": 2}
df["social_interaction_level"] = df["social_interaction_level"].map(social_map)

# addiction_level (1-10) → 3 კლასი: Mild / Moderate / Severe
def label_addiction(val):
    if val <= 3:
        return "Mild"
    elif val <= 6:
        return "Moderate"
    else:
        return "Severe"

df["addiction_class"] = df["addiction_level"].apply(label_addiction)

# target: ტექსტი → რიცხვი (0=Mild, 1=Moderate, 2=Severe)
label_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
df["target"] = df["addiction_class"].map(label_map)

# ── 3. X და y ─────────────────────────────────────────────────────────────────
features = [
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "stress_level",
    "social_interaction_level"
]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. MLflow ─────────────────────────────────────────────────────────────────
mlflow.set_experiment("smartphone_addiction")

with mlflow.start_run():

    max_iter = 1000
    solver   = "lbfgs"

    model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("solver",   solver)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
    print("MLflow run დასრულდა!")

# ── 5. pickle ─────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# label_map შენახვა (FastAPI-სთვის)
with open("label_map.pkl", "wb") as f:
    pickle.dump({0: "Mild", 1: "Moderate", 2: "Severe"}, f)

print("model.pkl და label_map.pkl შეინახა!")