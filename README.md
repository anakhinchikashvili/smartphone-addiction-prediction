# 📱 Smartphone Addiction Prediction

Machine Learning პროექტი, რომელიც პროგნოზირებს მოზარდების სმარტფონზე დამოკიდებულების დონეს სოციალური მედიის გამოყენებისა და ცხოვრების წესთან დაკავშირებული მონაცემების საფუძველზე.

## 🎯 Project Overview

მოდელი კლასიფიცირებს მომხმარებლებს შემდეგ კატეგორიებად:

* **Mild** – დაბალი დამოკიდებულება
* **Moderate** – საშუალო დამოკიდებულება
* **Severe** – მაღალი დამოკიდებულება

პროგნოზი ეფუძნება სოციალურ და ქცევით მახასიათებლებს, როგორიცაა სოციალური მედიის გამოყენების ხანგრძლივობა, ძილის დრო და სტრესის დონე.

---

## 📊 Dataset

**Source:** Teen Mental Health Dataset

**Records:** 1200

### Target Variable

| Addiction Score | Category |
| --------------- | -------- |
| 1 – 3           | Mild     |
| 4 – 6           | Moderate |
| 7 – 10          | Severe   |

### Features

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| daily_social_media_hours | სოციალური მედიის ყოველდღიური გამოყენების საათები |
| sleep_hours              | ძილის ხანგრძლივობა                               |
| screen_time_before_sleep | ეკრანთან გატარებული დრო ძილის წინ                |
| stress_level             | სტრესის დონე                                     |
| social_interaction_level | სოციალური აქტივობის დონე                         |

---

## 🏗️ Project Structure

```text
smartphone_addiction_project/
│
├── train.py                     # Model training and MLflow tracking
├── main.py                      # FastAPI application
├── requirements.txt             # Project dependencies
├── model.pkl                    # Trained model
├── label_map.pkl                # Label encoder mapping
└── Teen_Mental_Health_Dataset.csv
```

---

## ⚙️ Installation

### 1. Create Virtual Environment

```bash
python -m venv myenv
```

### 2. Activate Environment

**Windows**

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
* Trains a Logistic Regression model
* Saves the trained model
* Logs experiments using MLflow

---

## 📈 MLflow Tracking

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

MLflow tracks:

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

**POST** `/predict`

```json
{
  "daily_social_media_hours": 8.5,
  "sleep_hours": 5.0,
  "screen_time_before_sleep": 3.0,
  "stress_level": 8,
  "social_interaction_level": 0
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

| Parameter | Value               |
| --------- | ------------------- |
| Algorithm | Logistic Regression |
| Solver    | lbfgs               |
| max_iter  | 1000                |
| Accuracy  | 31.25%              |

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

---

## 👨‍💻 Author

Developed as a Machine Learning project for smartphone addiction level prediction using behavioral and social media usage data.
