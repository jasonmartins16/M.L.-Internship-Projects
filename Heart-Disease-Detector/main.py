from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

#Load the model and scaler
model = joblib.load("E:\Heart Disease\heart_disease_predictor.pkl")
scaler = joblib.load("E:\Heart Disease\scaler.pkl")

#Create FastApi
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Define the request schema
class PatientInfo(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_bps: float
    cholesterol: float
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: float
    exercise_angina: int
    oldpeak: float
    ST_slope: int

@app.post("/predict")
def predict(data: PatientInfo):
    # Convert the data to an array
    input_arr = np.array([[data.age, data.sex, data.chest_pain_type, data.resting_bps,
                             data.cholesterol, data.fasting_blood_sugar, data.resting_ecg,
                             data.max_heart_rate, data.exercise_angina, data.oldpeak, data.ST_slope]])

    # Scale the data
    scaled_data = scaler.transform(input_arr)

    # Get prediction probability
    prob = model.predict_proba(scaled_data)[0][1]

    # Apply custom threshold
    threshold = 0.4
    prediction = 1 if prob >= threshold else 0

    #predict 
    return {
        "prediction": prediction,
        "probability": round(float(prob), 3),
        "threshold": threshold
    }
    #prediction = model.predict(scaled_data)[0]
    #return {"prediction": int(prediction)}