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
