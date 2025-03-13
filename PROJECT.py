import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# อ่านข้อมูลจากไฟล์ Excel
file_path = r'C:\Users\acer\Desktop\IS\project\gold_market_data_2023_2025_corrected.xlsx'
df = pd.read_excel(file_path)

# แปลงคอลัมน์ 'วันที่' ให้เป็นประเภท DateTime โดยใช้ dayfirst=True
df['วันที่'] = pd.to_datetime(df['วันที่'], dayfirst=True)

# การสร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่ (Moving Averages)
df['MA_7'] = df['ราคา ทองคำ (บาท)'].rolling(window=7).mean()
df['MA_30'] = df['ราคา ทองคำ (บาท)'].rolling(window=30).mean()

# เลือกคอลัมน์ที่ใช้สำหรับการพัฒนาโมเดล
df = df[['วันที่', 'ราคา ทองคำ (บาท)', 'ปริมาณการซื้อขาย (กรัม)', 'อัตราแลกเปลี่ยน (USD/THB)', 'MA_7', 'MA_30']]

# เตรียมข้อมูล X (features) และ y (target)
X = df[['ปริมาณการซื้อขาย (กรัม)', 'อัตราแลกเปลี่ยน (USD/THB)', 'MA_7', 'MA_30']]
y = df['ราคา ทองคำ (บาท)']

# แบ่งข้อมูลเป็นชุดฝึกสอน (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# การทำ Grid Search สำหรับ Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

rf_model = RandomForestRegressor(random_state=42)

# ทำการค้นหาพารามิเตอร์ที่ดีที่สุด
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)

# การใช้พารามิเตอร์ที่ดีที่สุด
best_rf_model = rf_grid_search.best_estimator_

# การทำ Grid Search สำหรับ XGBoost
xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200],
    'colsample_bytree': [0.3, 0.5, 0.7],
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# ทำการค้นหาพารามิเตอร์ที่ดีที่สุด
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='neg_mean_squared_error')
xgb_grid_search.fit(X_train, y_train)

# การใช้พารามิเตอร์ที่ดีที่สุด
best_xgb_model = xgb_grid_search.best_estimator_

# เลือกโมเดลที่ดีที่สุดจาก R²
y_pred_rf = best_rf_model.predict(X_test)
y_pred_xgb = best_xgb_model.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)

if r2_rf > r2_xgb:
    best_model = best_rf_model
else:
    best_model = best_xgb_model

# ฟังก์ชันรับข้อมูลวันเดือนปีและพยากรณ์ราคาทองคำ
def predict_gold_price(date_input):
    # แปลงวันที่จากผู้ใช้เป็น DateTime
    date_input = pd.to_datetime(date_input, dayfirst=True)

    # หาเฉลี่ยเคลื่อนที่สำหรับวันนั้น (MA_7, MA_30)
    ma_7 = df.loc[df['วันที่'] <= date_input, 'ราคา ทองคำ (บาท)'].rolling(window=7).mean().iloc[-1]
    ma_30 = df.loc[df['วันที่'] <= date_input, 'ราคา ทองคำ (บาท)'].rolling(window=30).mean().iloc[-1]

    # ข้อมูลฟีเจอร์สำหรับวันนั้น (ต้องมีข้อมูลอื่นๆ เช่น ปริมาณการซื้อขายและอัตราแลกเปลี่ยน)
    latest_data = df.iloc[-1]
    X_input = pd.DataFrame([[latest_data['ปริมาณการซื้อขาย (กรัม)'], latest_data['อัตราแลกเปลี่ยน (USD/THB)'], ma_7, ma_30]],
                           columns=['ปริมาณการซื้อขาย (กรัม)', 'อัตราแลกเปลี่ยน (USD/THB)', 'MA_7', 'MA_30'])

    # พยากรณ์ราคาทองคำจากโมเดลที่ดีที่สุด
    predicted_price = best_model.predict(X_input)[0]
    return predicted_price

# สร้าง UI สำหรับ Streamlit
st.title('การพยากรณ์ราคาทองคำ')

# รับข้อมูลวันที่จากผู้ใช้
date_input = st.text_input("กรุณาใส่วันที่ (ในรูปแบบ dd/mm/yyyy):")

# เมื่อกดปุ่มพยากรณ์
if st.button('พยากรณ์ราคาทองคำ'):
    if date_input:
        predicted_price = predict_gold_price(date_input)
        st.write(f"ราคาทองคำที่พยากรณ์ในวันที่ {date_input} คือ: {predicted_price:.2f} บาท")
    else:
        st.write("กรุณากรอกวันที่ที่ต้องการพยากรณ์")

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# โหลดข้อมูลและเตรียมโมเดล
data = pd.read_csv('C:/Users/acer/Desktop/IS/project/temp.csv')  # ใช้เส้นทางไฟล์ที่ถูกต้อง

# แทนที่ค่า NaN ใน features ด้วย SimpleImputer
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(data[['Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin', 'LDAPS_RHmax', 
                                       'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS', 'Solar radiation', 
                                       'lat', 'lon', 'DEM', 'Slope']])

# แทนที่ค่า NaN ใน target variables (Next_Tmax และ Next_Tmin)
imputer_target = SimpleImputer(strategy='mean')
target_tmax = imputer_target.fit_transform(data['Next_Tmax'].values.reshape(-1, 1)).flatten()
target_tmin = imputer_target.fit_transform(data['Next_Tmin'].values.reshape(-1, 1)).flatten()

# แบ่งข้อมูลเป็นชุดฝึกและทดสอบ
X_train, X_test, y_train_tmax, y_test_tmax = train_test_split(features, target_tmax, test_size=0.2, random_state=42)
X_train, X_test, y_train_tmin, y_test_tmin = train_test_split(features, target_tmin, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล (Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล Random Forest Regressor
model_tmax = RandomForestRegressor(n_estimators=100, random_state=42)
model_tmin = RandomForestRegressor(n_estimators=100, random_state=42)

# ฝึกโมเดล
model_tmax.fit(X_train, y_train_tmax)
model_tmin.fit(X_train, y_train_tmin)

# สร้างหน้าแสดงผล
st.title('ทำนายอุณหภูมิสูงสุดและต่ำสุด')

# รับข้อมูลจากผู้ใช้ (วัน/เดือน/ปี)
st.subheader('กรอกข้อมูลวัน/เดือน/ปี:')
day = st.number_input('วัน', min_value=1, max_value=31, value=15)
month = st.number_input('เดือน', min_value=1, max_value=12, value=3)
year = st.number_input('ปี', min_value=2020, max_value=2025, value=2025)

# รับข้อมูลอื่น ๆ สำหรับการทำนาย (ใช้ค่าตัวอย่างสำหรับฟีเจอร์ที่เหลือ)
present_tmax = st.number_input('Present_Tmax (อุณหภูมิสูงสุดของวันปัจจุบัน)', value=20)
present_tmin = st.number_input('Present_Tmin (อุณหภูมิต่ำสุดของวันปัจจุบัน)', value=15)
rhmin = st.number_input('LDAPS_RHmin (ค่าความชื้นขั้นต่ำ)', value=65)
rhmax = st.number_input('LDAPS_RHmax (ค่าความชื้นสูงสุด)', value=85)
tmax_lapse = st.number_input('LDAPS_Tmax_lapse (อุณหภูมิสูงสุดล่วงหน้า)', value=30)
tmin_lapse = st.number_input('LDAPS_Tmin_lapse (อุณหภูมิต่ำสุดล่วงหน้า)', value=24)
ws = st.number_input('LDAPS_WS (ความเร็วลม)', value=5)
solar_radiation = st.number_input('Solar Radiation (รังสีจากดวงอาทิตย์)', value=500)
lat = st.number_input('Latitude (ละติจูด)', value=37.6046)
lon = st.number_input('Longitude (ลองจิจูด)', value=126.991)
dem = st.number_input('DEM (ความสูงเหนือระดับน้ำทะเล)', value=212)
slope = st.number_input('Slope (ความลาดชัน)', value=2.785)

# ข้อมูลที่ใช้ในการทำนาย
user_input = [[present_tmax, present_tmin, rhmin, rhmax, tmax_lapse, tmin_lapse, ws, solar_radiation, lat, lon, dem, slope]]

# ปรับข้อมูลให้พร้อมสำหรับการทำนาย
user_input_scaled = scaler.transform(user_input)

# ทำนายอุณหภูมิสูงสุดและต่ำสุด
predicted_tmax = model_tmax.predict(user_input_scaled)
predicted_tmin = model_tmin.predict(user_input_scaled)

# แสดงผลลัพธ์
if st.button('ทำนาย'):
    st.write(f"\nผลการทำนายสำหรับวันที่ {day}/{month}/{year}:")
    st.write(f"อุณหภูมิสูงสุด (Next_Tmax) ที่พยากรณ์: {predicted_tmax[0]}°C")
    st.write(f"อุณหภูมิขั้นต่ำ (Next_Tmin) ที่พยากรณ์: {predicted_tmin[0]}°C")
