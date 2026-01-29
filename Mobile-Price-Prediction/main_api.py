from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#import numpy as np
import pandas as pd
import joblib


#load the model
model = joblib.load("xgb_model.pkl")

app = FastAPI()

# Allow all origins (for development only — restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:5500"] or similar if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#define input features
class MobileFeatures(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int
    
@app.post("/predict")
def predict_price_category(features: MobileFeatures):
    
    #convert input data into a dataframe
    data = pd.DataFrame([features.model_dump()])                
    category_map = {
    0: "low",
    1: "medium",
    2: "high",
    3: "very high"
    }
    
    #predict
    prediction = model.predict(data)[0]

    return {"price_category": category_map[int(prediction)]}