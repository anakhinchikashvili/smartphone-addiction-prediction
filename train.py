import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── 1. მონაცემების ჩატვირთვა ──────────────────────────────────────────────────
df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")

# None/NaN მქონე სტრიქონები ამოვიღოთ (819 ცალი)
df = df.dropna(subset=['addiction_level'])

# ── 2. Feature Engineering (წინა დავალების იგივე) ────────────────────────────
# passive_usage: სოციალური მედია + გეიმინგი ჯამში
df['passive_usage'] = df['social_media_hours'] + df['gaming_hours']

# stress_level: ტექსტი → რიცხვი
stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['stress_encoded'] = df['stress_level'].map(stress_map)

# ── 3. Target ─────────────────────────────────────────────────────────────────
# addiction_level უკვე Mild/Moderate/Severe-ია — პირდაპირ გამოვიყენოთ
le = LabelEncoder()
df['target'] = le.fit_transform(df['addiction_level'])
# LabelEncoder: Mild=0, Moderate=1, Severe=2

# ── 4. Features და Train/Test Split ──────────────────────────────────────────
features = [
    'daily_screen_time_hours',
    'sleep_hours',
    'passive_usage',
    'stress_encoded',
    'app_opens_per_day'
]

X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (წინა დავალების იგივე)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. MLflow ─────────────────────────────────────────────────────────────────
mlflow.set_experiment("smartphone_addiction")

with mlflow.start_run():

    max_iter = 1000
    solver   = "lbfgs"

    model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight='balanced'
    )
    model.fit(X_train_sc, y_train)

    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)

    # MLflow-ში შენახვა
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("solver",   solver)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_))
    print("MLflow run დასრულდა!")

# ── 6. Pickle შენახვა (FastAPI-სთვის) ────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nmodel.pkl, scaler.pkl, label_encoder.pkl შეინახა!")