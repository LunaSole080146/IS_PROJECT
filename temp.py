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
