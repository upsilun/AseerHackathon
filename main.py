from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# تحميل النماذج المحفوظة
model_fire = joblib.load("model_fire.pkl")
model_air_quality = joblib.load("model_air_quality.pkl")
model_temperature = joblib.load("model_temperature.pkl")
model_soil_fertility = joblib.load("model_soil_fertility.pkl")

# ====== موديل حريق ======
class FireInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    air_quality_index: float

@app.post("/predict/fire")
def predict_fire(data: FireInput):
    features = np.array([[data.temperature, data.humidity, data.wind_speed, data.air_quality_index]])
    prediction = model_fire.predict(features)[0]
    return {"fire_detected": bool(prediction)}

# ====== موديل جودة الهواء ======
class AirQualityInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float

@app.post("/predict/air_quality")
def predict_air_quality(data: AirQualityInput):
    features = np.array([[data.temperature, data.humidity, data.wind_speed]])
    prediction = model_air_quality.predict(features)[0]
    return {"air_quality_index": float(prediction)}

# ====== موديل درجة الحرارة ======
class TemperatureInput(BaseModel):
    humidity: float
    wind_speed: float
    air_quality_index: float

@app.post("/predict/temperature")
def predict_temperature(data: TemperatureInput):
    features = np.array([[data.humidity, data.wind_speed, data.air_quality_index]])
    prediction = model_temperature.predict(features)[0]
    return {"temperature": float(prediction)}

# ====== موديل خصوبة التربة ======
class SoilFertilityInput(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float

@app.post("/predict/soil_fertility")
def predict_soil_fertility(data: SoilFertilityInput):
    features = np.array([[data.temperature, data.humidity, data.soil_moisture]])
    prediction = model_soil_fertility.predict(features)[0]
    return {"soil_fertility": float(prediction)}

