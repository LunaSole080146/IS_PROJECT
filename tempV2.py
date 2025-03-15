import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from scipy.stats import randint
import joblib

# โหลดข้อมูลจากไฟล์ Excel
file_path = r'C:\Users\acer\Desktop\IS\project\synthetic_temperature_data_2023.xlsx'
data = pd.read_excel(file_path)

# แปลงคอลัมน์ 'Date' ให้เป็นชนิดข้อมูล datetime
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# เพิ่มฟีเจอร์ใหม่ เช่น ฤดูกาล (Season), วันที่ในปี (DayOfYear), เดือน (Month)
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear

# แปลงค่าฤดูกาล (Season) ให้เป็นตัวเลข (Label Encoding)
season_encoder = LabelEncoder()
data['Season'] = season_encoder.fit_transform(data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                                                 else ('Spring' if x in [3, 4, 5] 
                                                                       else ('Summer' if x in [6, 7, 8] else 'Fall'))))

# เตรียมข้อมูล (ใช้ฟีเจอร์ที่เกี่ยวข้อง)
X = data[['Humidity', 'WindSpeed', 'SolarRadiation', 'Latitude', 'Longitude', 'Pressure', 'DayOfYear', 'Month', 'Season']]  # ฟีเจอร์
y = data['Temperature']  # เป้าหมาย: Temperature

# แบ่งข้อมูลเป็น training และ testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# เติมค่า NaN ใน X ด้วยค่าเฉลี่ยของแต่ละคอลัมน์ (ถ้ามี)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ปรับข้อมูลให้มีมาตรฐาน (Standardize)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# ตั้งค่าพารามิเตอร์ที่ต้องการใช้ในการค้นหา (RandomizedSearchCV)
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2]
}

# ใช้ RandomizedSearchCV เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=100, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# เลือกโมเดลที่ดีที่สุดจาก RandomizedSearchCV
best_xgb_model = random_search.best_estimator_

# บันทึกโมเดลที่ฝึกเสร็จแล้วและ StandardScaler
joblib.dump(best_xgb_model, 'best_xgb_model_temp.pkl')
joblib.dump(scaler, 'scaler_temp.pkl')  # บันทึก StandardScaler

# ฟังก์ชันทำนายอุณหภูมิสำหรับวันที่ต้องการ
def predict_temperature(input_date):
    # แปลงวันที่ที่ผู้ใช้ป้อนให้เป็น datetime โดยใช้รูปแบบ DD/MM/YYYY
    input_date = pd.to_datetime(input_date, format='%d/%m/%Y')
    day_of_year = input_date.dayofyear  # กำหนดค่า day_of_year จาก input_date
    
    # ใช้ค่าเฉลี่ยของฟีเจอร์ที่ไม่ใช่วันที่ (เพื่อใช้ในการทำนาย)
    input_data = [[data['Humidity'].mean(), data['WindSpeed'].mean(), data['SolarRadiation'].mean(),
                   data['Latitude'].mean(), data['Longitude'].mean(), data['Pressure'].mean(),
                   day_of_year, input_date.month, input_date.month]]  # ใช้หมายเลขเดือนแทน strftime('%b')

    # ปรับข้อมูลให้มีมาตรฐาน (standardize) ก่อนทำนาย
    input_data_scaled = scaler.transform(input_data)

    # ทำนายอุณหภูมิ
    pred_temperature = best_xgb_model.predict(input_data_scaled)[0]

    # แสดงผลวันที่ในรูปแบบ วัน-เดือน-ปี
    formatted_date = input_date.strftime('%d-%m-%Y')
    
    return formatted_date, pred_temperature

# รับ input วันที่จากผู้ใช้ในรูปแบบ DD/MM/YYYY
input_date = input("Enter the date you want to predict (DD/MM/YYYY): ")

# ทำนายอุณหภูมิสำหรับวันที่ที่ป้อน
formatted_date, pred_temperature = predict_temperature(input_date)

# แสดงผล
print(f"Predicted Temperature for {formatted_date}: {pred_temperature}°C")

# แสดงกราฟแท่งสำหรับ Tmax และ Tmin
plt.figure(figsize=(8, 6))
plt.bar([formatted_date], [pred_temperature], color=['green'])
plt.title(f"Temperature Prediction for {formatted_date}")
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
