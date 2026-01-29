from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("predict_survival.pkl")

# Initialize app
app = FastAPI(title="Lung Cancer Survival Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class PatientData(BaseModel):
    age: int
    gender: str
    cancer_stage: str
    smoking_status: str
    bmi: float
    cholesterol_level: float
    hypertension: str
    asthma: str
    cirrhosis: str
    other_cancer: str
    family_history: str
    treatment_type: str
    treatment_duration: int

@app.post("/predict")
def predict_survival(data: PatientData):
    # Normalize inputs (case-insensitive)
    gender = data.gender.strip().lower()
    cancer_stage = data.cancer_stage.strip().title()       # Stage I
    smoking_status = data.smoking_status.strip().title()   # Current Smoker
    treatment_type = data.treatment_type.strip().capitalize()  # Surgery

    # Mapping dictionaries
    gender_map = {'male': 0, 'female': 1}
    cancer_stage_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
    smoking_map = {
        'Never Smoked': 0,
        'Former Smoker': 1,
        'Passive Smoker': 2,
        'Current Smoker': 3
    }
    treatment_map = {'Chemotherapy': 0, 'Surgery': 1, 'Radiation': 2, 'Combined': 3}

    # Validate keys
    if gender not in gender_map or cancer_stage not in cancer_stage_map or smoking_status not in smoking_map or treatment_type not in treatment_map:
        return {"error": "Invalid input. Please check your form values."}

    # Prepare data
    input_data = pd.DataFrame([{
        'age': data.age,
        'gender': gender_map[gender],
        'cancer_stage': cancer_stage_map[cancer_stage],
        'smoking_status': smoking_map[smoking_status],
        'bmi': np.clip(data.bmi, 10, 50),
        'cholesterol_level': np.clip(data.cholesterol_level, 100, 350),
        'hypertension': 1 if data.hypertension.lower() == 'yes' else 0,
        'asthma': 1 if data.asthma.lower() == 'yes' else 0,
        'cirrhosis': 1 if data.cirrhosis.lower() == 'yes' else 0,
        'other_cancer': 1 if data.other_cancer.lower() == 'yes' else 0,
        'family_history': 1 if data.family_history.lower() == 'yes' else 0,
        'treatment_type': treatment_map[treatment_type],
        'treatment_duration': data.treatment_duration
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].tolist()

    return {
        "survived": "Yes" if prediction == 1 else "No",
        "probability": {"No": round(probability[0], 4), "Yes": round(probability[1], 4)}
    }
