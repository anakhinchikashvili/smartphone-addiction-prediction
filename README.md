# 📱 Smartphone Addiction Prediction

Machine Learning პროექტი, რომელიც პროგნოზირებს მომხმარებლის სმარტფონზე დამოკიდებულების დონეს სმარტფონის გამოყენების ჩვევებისა და ქცევითი მონაცემების საფუძველზე.

## 🎯 Project Overview

მოდელი კლასიფიცირებს მომხმარებლებს შემდეგ კატეგორიებად:

* **Mild** – მსუბუქი დამოკიდებულება
* **Moderate** – საშუალო დამოკიდებულება
* **Severe** – მძიმე დამოკიდებულება

პროგნოზი ეფუძნება მომხმარებლის ეკრანთან გატარებულ დროს, ძილის ხანგრძლივობას, პასიურ გამოყენებას, სტრესის დონესა და აპლიკაციების გამოყენების ინტენსივობას.

---

## 📊 Dataset

**Source:** Smartphone Usage And Addiction Analysis Dataset

**Records:** 7500

### Target Variable

| Addiction Level | Category         |
| --------------- | ---------------- |
| Mild            | Low Addiction    |
| Moderate        | Medium Addiction |
| Severe          | High Addiction   |

### Features

| Feature                   | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `daily_screen_time_hours` | ეკრანთან გატარებული დრო დღეში (საათებში)         |
| `sleep_hours`             | ძილის ხანგრძლივობა                               |
| `passive_usage`           | სოციალური მედიისა და გეიმინგის ჯამური გამოყენება |
| `stress_encoded`          | სტრესის დონე (0 = Low, 1 = Medium, 2 = High)     |
| `app_opens_per_day`       | აპლიკაციების გახსნის რაოდენობა დღეში             |

---

## 🏗️ Project Structure

```text
smartphone_addiction_project/
│
├── train.py
├── main.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── label_encoder.pkl
└── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
```

### Files Description

| File                | Purpose                            |
| ------------------- | ---------------------------------- |
| `train.py`          | Model training and MLflow tracking |
| `main.py`           | FastAPI application                |
| `model.pkl`         | Trained Logistic Regression model  |
| `scaler.pkl`        | StandardScaler object              |
| `label_encoder.pkl` | Label encoder                      |
| `requirements.txt`  | Project dependencies               |

---

## ⚙️ Installation

### 1. Create Virtual Environment

```bash
python -m venv myenv
```

### 2. Activate Environment

Windows:

```bash
myenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Model Training

Run the training script:

```bash
python train.py
```

This process:

* Loads and preprocesses the dataset
* Applies feature scaling using StandardScaler
* Trains a Logistic Regression model
* Saves the trained model and preprocessing artifacts
* Logs experiments using MLflow

---

## 📈 MLflow Tracking

Start MLflow UI:

```bash
python -m mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

### MLflow Tracks

* Parameters
* Metrics
* Accuracy
* Model artifacts

---

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🔮 Prediction Endpoint

### Request

**POST /predict**

```json
{
  "daily_screen_time_hours": 8.5,
  "sleep_hours": 5.0,
  "passive_usage": 4.2,
  "stress_encoded": 2,
  "app_opens_per_day": 120
}
```

### Response

```json
{
  "prediction": "Severe"
}
```

---

## 🧠 Machine Learning Model

| Parameter    | Value               |
| ------------ | ------------------- |
| Algorithm    | Logistic Regression |
| Solver       | lbfgs               |
| max_iter     | 1000                |
| class_weight | balanced            |
| Accuracy     | 51.38%              |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* MLflow

---

## 📌 Future Improvements

* Feature engineering
* Hyperparameter tuning
* Cross-validation
* Advanced classification models (Random Forest, XGBoost)
* Improved prediction accuracy
* Docker deployment

---

## 👩‍💻 Author

Developed as a Machine Learning project for smartphone addiction level prediction using smartphone usage and behavioral data.
