import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# หน้า 1: การอธิบายแนวทางการพัฒนาโมเดลการทำนายราคาทองคำ
def gold_price_development():
    st.title("การพัฒนาโมเดลการทำนายราคาทองคำ")

    # การเตรียมข้อมูล
    st.header("การเตรียมข้อมูล")
    st.write("""
    ข้อมูลที่ใช้ในการพัฒนาโมเดลนี้ได้แก่ gold_market_data_2023_2025_corrected.xlsx ซึ่งใช้สำหรับทำนายราคาทองคำ ข้อมูลนี้ได้ถูกทำความสะอาด 
    และเลือกฟีเจอร์ที่สำคัญ เช่น ราคาทองคำ, ปริมาณการซื้อขาย, อัตราแลกเปลี่ยน
    """)

    # ทฤษฎีของอัลกอริธึมที่พัฒนา
    st.header("ทฤษฎีของอัลกอริธึมที่พัฒนา")
    st.write("""
    - **Random Forest**: ใช้หลักการของการสร้างหลายๆ ต้นไม้ (decision trees) แล้วทำการรวมผลลัพธ์จากทุกต้นไม้ เพื่อให้ได้ผลลัพธ์ที่ดีที่สุด
    - **XGBoost**: ใช้หลักการของ Gradient Boosting ซึ่งเป็นการพัฒนาโมเดลที่มีประสิทธิภาพสูง โดยการปรับแต่งพารามิเตอร์ให้เหมาะสม
    """)

    # ขั้นตอนการพัฒนาโมเดล
    st.header("ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. การเตรียมข้อมูลจากแหล่งข้อมูลที่เกี่ยวข้อง
    2. การแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    3. การใช้ Grid Search เพื่อหาพารามิเตอร์ที่ดีที่สุดสำหรับโมเดล
    4. การฝึกโมเดลและทดสอบประสิทธิภาพของโมเดล
    5. การเลือกโมเดลที่มีค่า R² สูงที่สุด
    """)

# หน้า 2: Demo การทำงานของโมเดลการทำนายราคาทองคำ
def gold_price_demo():
    st.title("การทดสอบโมเดลการทำนายราคาทองคำ")
    
    # ฟังก์ชันทำนายราคาทองคำ
    def predict_gold_price(date_input):
        # แก้ไขเส้นทางไฟล์ให้ตรงกับที่ไฟล์ถูกอัปโหลดใน Streamlit Cloud
        file_path = '/mnt/data/gold_market_data_2023_2025_corrected.xlsx'
        df = pd.read_excel(file_path)
        
        df['วันที่'] = pd.to_datetime(df['วันที่'], dayfirst=True)
        df['MA_7'] = df['ราคา ทองคำ (บาท)'].rolling(window=7).mean()
        df['MA_30'] = df['ราคา ทองคำ (บาท)'].rolling(window=30).mean()
        df = df[['วันที่', 'ราคา ทองคำ (บาท)', 'ปริมาณการซื้อขาย (กรัม)', 'อัตราแลกเปลี่ยน (USD/THB)', 'MA_7', 'MA_30']]

        X = df[['ปริมาณการซื้อขาย (กรัม)', 'อัตราแลกเปลี่ยน (USD/THB)', 'MA_7', 'MA_30']]
        y = df['ราคา ทองคำ (บาท)']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)

        # ทำนายราคาทองคำ
        date_input = pd.to_datetime(date_input, dayfirst=True)
        ma_7 = df.loc[df['วันที่'] <= date_input, 'ราคา ทองคำ (บาท)'].rolling(window=7).mean().iloc[-1]
        ma_30 = df.loc[df['วันที่'] <= date_input, 'ราคา ทองคำ (บาท)'].rolling(window=30).mean().iloc[-1]
        latest_data = df.iloc[-1]
        X_input = pd.DataFrame([[latest_data['ปริมาณการซื้อขาย (กรัม)'], latest_data['อัตราแลกเปลี่ยน (USD/THB)'], ma_7, ma_30]],
                               columns=['ปริมาณการซื้อขาย (กรัม)', 'อัตราแลกเปลี่ยน (USD/THB)', 'MA_7', 'MA_30'])
        predicted_price = rf_model.predict(X_input)[0]
        return predicted_price
    
    date_input = st.text_input("กรุณาใส่วันที่ (dd/mm/yyyy):")
    if st.button('พยากรณ์ราคาทองคำ'):
        if date_input:
            predicted_price = predict_gold_price(date_input)
            st.write(f"ราคาทองคำที่พยากรณ์ในวันที่ {date_input}: {predicted_price:.2f} บาท")
        else:
            st.write("กรุณากรอกวันที่ที่ต้องการพยากรณ์")

# หน้า 3: การอธิบายแนวทางการพัฒนาโมเดลการทำนายอุณหภูมิ
def temperature_development():
    st.title("การพัฒนาโมเดลการทำนายอุณหภูมิ")

    # การเตรียมข้อมูล
    st.header("การเตรียมข้อมูล")
    st.write("""
    ข้อมูลที่ใช้ในการพัฒนาโมเดลนี้ได้แก่ temp.csv ซึ่งใช้สำหรับทำนายอุณหภูมิ ข้อมูลนี้ถูกทำความสะอาดและเลือกฟีเจอร์ที่สำคัญ
    เช่น อุณหภูมิสูงสุด, อุณหภูมิต่ำสุด, ความชื้น, ความเร็วลม และข้อมูลอื่นๆ ที่เกี่ยวข้อง
    """)

    # ทฤษฎีของอัลกอริธึมที่พัฒนา
    st.header("ทฤษฎีของอัลกอริธึมที่พัฒนา")
    st.write("""
    - **Random Forest**: การสร้างหลายๆ ต้นไม้แล้วนำผลลัพธ์จากแต่ละต้นไม้มารวมกันเพื่อให้ผลลัพธ์ที่ดีที่สุด
    - **XGBoost**: ใช้การพัฒนาของ Gradient Boosting เพื่อสร้างโมเดลที่มีประสิทธิภาพสูง
    """)

    # ขั้นตอนการพัฒนาโมเดล
    st.header("ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. การเตรียมข้อมูลจากไฟล์ temp.csv
    2. การแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    3. การปรับพารามิเตอร์โดยใช้ Grid Search
    4. การฝึกโมเดล Random Forest และ XGBoost
    5. การเลือกโมเดลที่ดีที่สุดจากค่าความแม่นยำ
    """)

# หน้า 4: Demo การทำงานของโมเดลการทำนายอุณหภูมิ
def temperature_demo():
    st.title("การทดสอบโมเดลการทำนายอุณหภูมิ")
    
    # ฟังก์ชันทำนายอุณหภูมิ
    def predict_temperature(features):
        # แก้ไขเส้นทางไฟล์ให้ตรงกับที่ไฟล์ถูกอัปโหลดใน Streamlit Cloud
        file_path = '/mnt/data/temp.csv'
        data = pd.read_csv(file_path)

        imputer = SimpleImputer(strategy='mean')
        features = im
