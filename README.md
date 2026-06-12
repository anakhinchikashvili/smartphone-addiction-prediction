# Smartphone Addiction Prediction

## პროექტის აღწერა
მოდელი პროგნოზირებს მოზარდების სმარტფონზე დამოკიდებულების დონეს (Mild / Moderate / Severe) სოციალური მედიის გამოყენების მონაცემების საფუძველზე.

## მონაცემები
- **Dataset:** Teen Mental Health Dataset (1200 ჩანაწერი)
- **Target:** addiction_level → Mild (1-3), Moderate (4-6), Severe (7-10)
- **Features:** daily_social_media_hours, sleep_hours, screen_time_before_sleep, stress_level, social_interaction_level

## პროექტის სტრუქტურა
smartphone_addiction_project/
├── train.py                     # მოდელის გაწვრთნა + MLflow
├── main.py                      # FastAPI სერვისი
├── requirements.txt             # საჭირო ბიბლიოთეკები
├── model.pkl                    # გაწვრთნილი მოდელი
├── label_map.pkl                # კლასების რუქა
└── Teen_Mental_Health_Dataset.csv

## გაშვება

### 1. გარემოს მომზადება
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt

### 2. მოდელის გაწვრთნა
python train.py

### 3. MLflow UI
mlflow ui
# ბრაუზერში: http://127.0.0.1:5000

### 4. API სერვერის გაშვება
uvicorn main:app --reload
# ბრაუზერში: http://127.0.0.1:8000/docs

## API გამოყენება

**POST /predict**

Request:
{
  "daily_social_media_hours": 8.5,
  "sleep_hours": 5.0,
  "screen_time_before_sleep": 3.0,
  "stress_level": 8,
  "social_interaction_level": 0
}

Response:
{
  "prediction": "Severe"
}

## მოდელი
- **ალგორითმი:** Logistic Regression
- **Solver:** lbfgs
- **max_iter:** 1000
- **Accuracy:** 31.25%