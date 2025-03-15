import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# โหลดโมเดลที่ฝึกแล้ว
model_temp = joblib.load('best_xgb_model_temp.pkl')
scaler_temp = joblib.load('scaler_temp.pkl')

model_gold = joblib.load('best_rf_model_gold.pkl')

# สร้างฟังก์ชันทำนายอุณหภูมิ
def predict_temperature(input_data):
    input_data_scaled = scaler_temp.transform(input_data)
    pred_temperature = model_temp.predict(input_data_scaled)[0]
    return pred_temperature

# สร้างฟังก์ชันทำนายราคาทองคำ
def predict_gold(input_data):
    pred_gold_price = model_gold.predict(input_data)[0]
    return pred_gold_price

# ส่วนของ Streamlit Web App
st.title('Demo: Temperature and Gold Price Prediction')
st.write("Welcome to the prediction app. This app allows you to predict **Temperature** and **Gold Price** based on input data. Let's begin by understanding how the models work.")

# คำอธิบายเกี่ยวกับโมเดลการทำนายอุณหภูมิ
st.subheader('Model 1: Temperature Prediction')
st.write("""
The **Temperature Prediction** model is based on the **XGBoost Regressor**, which is an ensemble machine learning model that utilizes boosting to make predictions. 
This model is trained using various features like **humidity**, **wind speed**, **solar radiation**, **latitude**, and **pressure** to predict the temperature for a given day.

The input data includes:
- **Humidity**: Percentage of moisture in the air.
- **Wind Speed**: Speed of the wind (km/h).
- **Solar Radiation**: Amount of solar radiation (W/m²).
- **Latitude**: The latitude of the location.
- **Longitude**: The longitude of the location.
- **Pressure**: Atmospheric pressure (hPa).
- **Day of the Year**: A numeric representation of the day in the year.
- **Month**: The month of the year.
""")

# รับข้อมูลจากผู้ใช้สำหรับการทำนายอุณหภูมิ
st.subheader('Enter details for Temperature Prediction')
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=70)
wind_speed = st.number_input('Wind Speed (km/h)', min_value=0, max_value=100, value=10)
solar_radiation = st.number_input('Solar Radiation (W/m²)', min_value=0, max_value=1000, value=200)
# เปลี่ยน min_value และ max_value ให้เป็น float
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=13.736)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=100.523)

pressure = st.number_input('Pressure (hPa)', min_value=900, max_value=1100, value=1013)
date = st.date_input('Date', value=pd.to_datetime('2023-03-15'))

# คำนวณ DayOfYear จากวันที่ที่ผู้ใช้ป้อน
day_of_year = pd.to_datetime(date).dayofyear

# สร้างข้อมูลที่ใช้ในการทำนายอุณหภูมิ
input_data = np.array([[humidity, wind_speed, solar_radiation, latitude, longitude, pressure, day_of_year, date.month, date.month]])

if st.button('Predict Temperature'):
    pred_temperature = predict_temperature(input_data)
    st.write(f'Predicted Temperature for {date} is {pred_temperature:.2f}°C')

# คำอธิบายเกี่ยวกับโมเดลการทำนายราคาทองคำ
st.subheader('Model 2: Gold Price Prediction')
st.write("""
The **Gold Price Prediction** model uses the **Random Forest Regressor**, which is a machine learning model that builds multiple decision trees and combines their predictions to make a final decision. 
This model is trained with features like **volume**, **market conditions**, and **exchange rate** to predict the price of gold.

The input data includes:
- **Volume**: The amount of gold in kilograms.
- **Market Condition**: The current market condition, either **Bullish** or **Bearish**.
- **Exchange Rate**: The exchange rate between USD and the local currency.
""")

# รับข้อมูลจากผู้ใช้สำหรับการทำนายราคาทองคำ
st.subheader('Enter details for Gold Price Prediction')
volume = st.number_input('Volume (kg)', min_value=0.0, max_value=1000.0, value=10.0)
market_condition = st.selectbox('Market Condition', ['Bullish', 'Bearish'])
exchange_rate = st.number_input('Exchange Rate (USD/Local)', min_value=0.5, max_value=1.5, value=1.0)

# สร้างข้อมูลที่ใช้ในการทำนายราคาทองคำ
input_data_gold = np.array([[volume, market_condition == 'Bullish', exchange_rate]])

if st.button('Predict Gold Price'):
    pred_gold_price = predict_gold(input_data_gold)
    st.write(f'Predicted Gold Price is {pred_gold_price:.2f} USD per kg')
