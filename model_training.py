import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import os
os.makedirs("models", exist_ok=True)

# توليد بيانات خصوبة التربة
data1 = pd.DataFrame({
    "temperature": np.random.uniform(15, 45, 1000),
    "humidity": np.random.uniform(10, 90, 1000),
    "soil_moisture": np.random.uniform(5, 80, 1000),
    "fertility": np.random.uniform(0, 100, 1000)
})

# تدريب خصوبة التربة
fertility_model = LinearRegression()
fertility_model.fit(data1[["temperature", "humidity", "soil_moisture"]], data1["fertility"])
joblib.dump(fertility_model, "models/fertility_model.pkl")

# بيانات التصحر
past_temp = np.random.uniform(20, 45, 1000)
past_humid = np.random.uniform(10, 50, 1000)
desert_now = np.random.uniform(0, 100, 1000)
desert_future = desert_now + np.random.uniform(-5, 10, 1000)

# تدريب التصحر
X_desert = pd.DataFrame({"past_temp": past_temp, "past_humid": past_humid, "now": desert_now})
y_desert = desert_future

desert_model = LinearRegression()
desert_model.fit(X_desert, y_desert)
joblib.dump(desert_model, "models/desertification_model.pkl")

# بيانات جودة الهواء
aqi_data = pd.DataFrame({
    "temperature": np.random.uniform(20, 50, 1000),
    "humidity": np.random.uniform(10, 80, 1000),
    "dust_level": np.random.uniform(0, 150, 1000),
    "aqi": np.random.uniform(0, 500, 1000)
})

air_model = LinearRegression()
air_model.fit(aqi_data[["temperature", "humidity", "dust_level"]], aqi_data["aqi"])
joblib.dump(air_model, "models/air_quality_model.pkl")

# بيانات الاقتراحات الزراعية
plants = ["عرعر", "طلح", "نخيل", "زيتون", "تين", "حمضيات"]

sugg_data = pd.DataFrame({
    "soil_moisture": np.random.uniform(10, 80, 1000),
    "fertility": np.random.uniform(0, 100, 1000),
    "ph": np.random.uniform(5, 9, 1000),
    "nitrogen": np.random.uniform(0, 100, 1000),
    "plant_type": [random.choice(plants) for _ in range(1000)],
    "days_until_best_time": np.random.randint(0, 180, 1000)
})

from sklearn.ensemble import RandomForestClassifier

plant_model = RandomForestClassifier()
plant_model.fit(sugg_data[["soil_moisture", "fertility", "ph", "nitrogen"]], sugg_data["plant_type"])

joblib.dump(plant_model, "models/suggestions_model.pkl")

print("✅ All models trained and saved!")

