import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def masuk_file(filename):
    try:
        df = pd.read_csv(filename)
        print(f"Berhasil memuat {filename}")
    except FileNotFoundError:
        print(f"File {filename} tidak ditemukan. Menggunakan data dummy.")
        data = {
            'area': [2600, 3000, 3200, 3600, 4000, 2500, 2700, 3100, 3300, 3700, 2800, 2900],
            'price': [550000, 565000, 610000, 680000, 725000, 540000, 560000, 600000, 620000, 690000, 570000, 580000]
        }
        df = pd.DataFrame(data)
    
    # Tentukan X dan y
    X = df[['area']]
    y = df['price']
    return X, y


def latih_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return slope, intercept, rmse, X_test, y_test, y_pred

def tampilkan_hasil_teks(slope, intercept, rmse, X_test, y_test, y_pred):
    print("HASIL REGRESI LINEAR")
    print(f"Persamaan: y = {slope:.2f}x + {intercept:.2f}")
    print(f"RMSE     : {rmse:.4f}")

    print("\n DATA LUAS")
    for val in X_test['area'].values:
        print(f"{val:.2f}")

    print("\n DATA HARGA PREDIKSI")
    for val in y_pred:
        print(f"{val:.2f}")

    print("\n DATA HARGA AKTUAL")
    for val in y_test.values:
        print(f"{val:.2f}")
    print("-" * 40)

def tampilkan_grafik(X_test, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(X_test, y_test, color='#2d5a8c', alpha=0.6, s=45, label='Data Aktual')
    ax.plot(X_test, y_pred, color='#c74440', linewidth=1.8, label='Garis Regresi')
    
    ax.set_title('Hubungan Luas Tanah terhadap Harga Rumah')
    ax.set_xlabel('Luas Tanah')
    ax.set_ylabel('Harga Rumah')
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    nama_file = 'data.csv'

    X, y = masuk_file(nama_file)
    
    slope, intercept, rmse, X_test, y_test, y_pred = latih_model(X, y)
    
    tampilkan_hasil_teks(slope, intercept, rmse, X_test, y_test, y_pred)

    tampilkan_grafik(X_test, y_test, y_pred)