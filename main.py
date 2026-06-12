from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# მოდელის ჩატვირთვა
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

app = FastAPI()

# შემავალი მონაცემების სქემა
class InputData(BaseModel):
    daily_social_media_hours: float
    sleep_hours: float
    screen_time_before_sleep: float
    stress_level: int
    social_interaction_level: int  # 0=low, 1=medium, 2=high

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[
        data.daily_social_media_hours,
        data.sleep_hours,
        data.screen_time_before_sleep,
        data.stress_level,
        data.social_interaction_level
    ]])

    pred_num   = model.predict(X)[0]
    pred_label = label_map[pred_num]

    return {"prediction": pred_label}