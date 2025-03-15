import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import numpy as np
from datetime import datetime
import joblib

# โหลดข้อมูลจากไฟล์ Excel
file_path = r'C:\Users\acer\Desktop\IS\project\golddataset.xlsx'
df = pd.read_excel(file_path)

# แปลง 'วันที่' เป็น datetime
df['วันที่'] = pd.to_datetime(df['วันที่'])

# แปลง 'สภาวะตลาด' เป็นตัวเลข (1 สำหรับ 'ขาขึ้น', 0 สำหรับ 'ขาลง')
df['สภาวะตลาด'] = df['สภาวะตลาด'].apply(lambda x: 1 if x == 'ขาขึ้น' else 0)

# เลือกฟีเจอร์และเป้าหมาย
features = ['ปริมาณการซื้อขาย (กรัม)', 'สภาวะตลาด', 'อัตราแลกเปลี่ยน (USD/THB)']
target = 'ราคา ทองคำ (บาท)'

# แบ่งข้อมูลเป็น X (ฟีเจอร์) และ y (เป้าหมาย)
X = df[features]
y = df[target]

# สร้างและฝึกโมเดล Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ประเมินผลการทำนายบนชุดข้อมูลทดสอบ
mae = mean_absolute_error(y, rf_model.predict(X))
mse = mean_squared_error(y, rf_model.predict(X))

# แสดงความแม่นยำ
accuracy = (1 - mae / y.mean()) * 100  # คำนวณความแม่นยำเป็นเปอร์เซ็นต์
print(f"Model Accuracy - MAE: {mae:.2f}, MSE: {mse:.2f}, Accuracy: {accuracy:.2f}%")

# รับ input วันเดือนปีจากผู้ใช้ (รูปแบบ DD/MM/YYYY)
input_date = input("Enter the date (DD/MM/YYYY): ")

# ลบอักขระพิเศษหรือช่องว่างที่ไม่ต้องการ (เช่น "ๅ" หรืออื่นๆ)
input_date = input_date.replace('ๅ', '').strip()  # ลบอักขระที่ไม่ต้องการ

# แปลงวันที่จาก input เป็น datetime
try:
    input_date = datetime.strptime(input_date, "%d/%m/%Y")
    print(f"Input date: {input_date.date()}")
except ValueError:
    print("Invalid date format. Please enter the date in DD/MM/YYYY format.")
    exit()  # Exit the program if invalid format is entered

# สร้างข้อมูลทำนายสำหรับ 30 วัน
future_dates = [input_date + timedelta(days=i) for i in range(1, 31)]  # 30 วันถัดไป

# สร้างฟีเจอร์สำหรับทำนาย (สามารถปรับแต่งให้เป็นค่าจริงจากผู้ใช้ได้)
future_data = pd.DataFrame({
    'ปริมาณการซื้อขาย (กรัม)': np.random.choice(df['ปริมาณการซื้อขาย (กรัม)'], size=30),
    'สภาวะตลาด': np.random.choice(df['สภาวะตลาด'], size=30),
    'อัตราแลกเปลี่ยน (USD/THB)': np.random.choice(df['อัตราแลกเปลี่ยน (USD/THB)'], size=30)
})

# ทำนายราคาทองคำจากข้อมูลที่ใส่
future_predictions = rf_model.predict(future_data)

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
joblib.dump(rf_model, 'best_rf_model_gold.pkl')

# แสดงผลลัพธ์การทำนาย
print(f"Predicted gold price on {input_date.date()}: {future_predictions[0]:.2f} THB")

# สร้างกราฟการทำนายในอนาคต
plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_predictions, color='green', marker='o', label='Predicted Gold Price')
plt.xlabel('Future Dates')
plt.ylabel('Predicted Gold Price (THB)')
plt.title('Gold Price Prediction for the Next 30 Days')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
