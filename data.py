import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. MEMUAT DATA
# Masukin nama file CSV/Excel nya disini
filename = 'data.csv'  # Ganti sesuai nama file data kamu
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"File {filename} tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

# Menentukan Variabel Independen (X) dan Dependen (y)
# X = Area (Luas Tanah), y = Price (Harga)
X = df[['area']] 
y = df['price']

# 2. SPLIT DATA (80% Training, 20% Testing)
# Sesuai standar soal UAS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MELATIH MODEL REGRESI
model = LinearRegression()
model.fit(X_train, y_train)

# Mendapatkan Slope (m) dan Intercept (c)
slope = model.coef_[0]
intercept = model.intercept_

# 4. MELAKUKAN PREDIKSI (Pada data testing)
y_pred = model.predict(X_test)

# 5. MENGHITUNG RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ==========================================
# OUTPUT HASIL (Disesuaikan dengan formatmu)
# ==========================================

print("=" * 40)
print("HASIL REGRESI LINEAR (HOUSING DATASET)")
print("=" * 40)

# Menampilkan Persamaan Tren
print(f"Persamaan Tren: y = {slope:.2f}x + {intercept:.2f}")
print(f"Nilai RMSE    : {rmse:.4f}")
print("-" * 40)

# Menampilkan 10 data perbandingan pertama (agar layar tidak penuh)
print("Sampel 10 Data Prediksi vs Aktual:")
# Menggabungkan data untuk tampilan
hasil = pd.DataFrame({'Luas (Area)': X_test['area'].values, 
                      'Harga Aktual': y_test.values, 
                      'Harga Prediksi': y_pred})

for index, row in hasil.head(10).iterrows():
    print(f"Luas: {int(row['Luas (Area)']):<5} | "
          f"Aktual: {int(row['Harga Aktual']):<10} | "
          f"Prediksi: {row['Harga Prediksi']:.2f}")

print("-" * 40)

# (Opsional) Visualisasi Grafik seperti "hasil teman"
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Data Aktual (Test Set)')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Garis Regresi')
plt.title('Regresi Linear: Luas Tanah vs Harga Rumah')
plt.xlabel('Luas Tanah (Area)')
plt.ylabel('Harga (Price)')
plt.legend()
plt.grid(True)
plt.show()