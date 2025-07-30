import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# إنشاء بيانات بسيطة للتجربة
data = {
    "temperature": [30, 45, 20, 50, 35, 25],
    "humidity": [10, 15, 60, 5, 20, 55],
    "wind_speed": [12, 25, 5, 30, 15, 8],
    "air_quality_index": [70, 90, 40, 95, 80, 35],
    "fire_detected": [1, 1, 0, 1, 1, 0]  # 1 = فيه حريق، 0 = مافيه
}

# نحولها لجدول
df = pd.DataFrame(data)

# نحدد المدخلات والمخرجات
X = df.drop("fire_detected", axis=1)
y = df["fire_detected"]

# نقسم البيانات لتدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# نبني نموذج ذكاء اصطناعي
model = RandomForestClassifier()
model.fit(X_train, y_train)

# نحفظ النموذج في ملف
joblib.dump(model, "model_fire.pkl")

print("✅ تم تدريب وحفظ النموذج بنجاح!")
