from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# ── მოდელების ჩატვირთვა ───────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = FastAPI()

# ── შემავალი მონაცემების სქემა ────────────────────────────────────────────────
class InputData(BaseModel):
    daily_screen_time_hours: float   # მაგ: 8.5
    sleep_hours: float               # მაგ: 5.0
    passive_usage: float             # social_media_hours + gaming_hours, მაგ: 4.2
    stress_encoded: int              # 0=Low, 1=Medium, 2=High
    app_opens_per_day: int           # მაგ: 120

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[
        data.daily_screen_time_hours,
        data.sleep_hours,
        data.passive_usage,
        data.stress_encoded,
        data.app_opens_per_day
    ]])

    X_scaled   = scaler.transform(X)
    pred_num   = model.predict(X_scaled)[0]
    pred_label = le.inverse_transform([pred_num])[0]

    return {"prediction": pred_label}