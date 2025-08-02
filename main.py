from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="نظام سدر - API")

fertility_model = joblib.load("models/fertility_model.pkl")
desert_model = joblib.load("models/desertification_model.pkl")
air_model = joblib.load("models/air_quality_model.pkl")
plant_model = joblib.load("models/suggestions_model.pkl")

# --- Schemas ---
class FertilityInput(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float

class DesertInput(BaseModel):
    past_temp: float
    past_humid: float
    now: float

class AirInput(BaseModel):
    temperature: float
    humidity: float
    dust_level: float

class SuggestionInput(BaseModel):
    soil_moisture: float
    fertility: float
    ph: float
    nitrogen: float

# --- Endpoints ---
@app.post("/predict/fertility")
def predict_fertility(data: FertilityInput):
    pred = fertility_model.predict([[data.temperature, data.humidity, data.soil_moisture]])
    return {"fertility_percentage": round(float(pred[0]), 2)}

@app.post("/predict/desertification")
def predict_desert(data: DesertInput):
    pred = desert_model.predict([[data.past_temp, data.past_humid, data.now]])
    return {"future_desertification": round(float(pred[0]), 2)}

@app.post("/predict/air_quality")
def predict_air(data: AirInput):
    pred = air_model.predict([[data.temperature, data.humidity, data.dust_level]])
    return {"aqi_estimation": round(float(pred[0]), 2)}

@app.post("/predict/agriculture_suggestion")
def predict_agri(data: SuggestionInput):
    plant_pred = plant_model.predict([[data.soil_moisture, data.fertility, data.ph, data.nitrogen]])[0]
    best_days = np.random.randint(0, 180)
    return {
        "recommended_plant": plant_pred,
        "days_until_best_time": best_days
    }
