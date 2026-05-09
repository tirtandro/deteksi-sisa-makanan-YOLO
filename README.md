# S3C-Smart-Canteen: Sistem Analisis Sisa Makanan

Sistem analisis sisa makanan berbasis Artificial Intelligence (AI) untuk aplikasi **S3C-Smart-Canteen**. Proyek ini menggunakan model YOLO11 untuk *instance segmentation* guna mendeteksi jenis sisa makanan dan mengestimasi beratnya (dalam gram) berdasarkan foto 2D ompreng setelah makan.

## 🚀 Fitur Utama
- **Deteksi Sisa Makanan**: Menggunakan YOLO11 Segmentation untuk identifikasi otomatis kategori makanan custom.
- **Estimasi Berat Akurat**: Algoritma matematis yang mengonversi luas piksel ke skala dunia nyata menggunakan kalibrasi objek referensi (ompreng standar).
- **API Backend**: Dibangun dengan Flask, siap diintegrasikan dengan frontend atau aplikasi mobile.
- **Dashboard Tester**: Antarmuka web dalam Bahasa Indonesia untuk pengujian fungsi deteksi secara real-time.
- **Kalibrasi Otomatis**: Mendeteksi ompreng dalam gambar untuk menentukan skala cm/pixel secara dinamis.

## 🛠️ Spesifikasi Teknis
- **Model AI**: YOLO11 Instance Segmentation (Ultralytics).
- **Kategori Terdeteksi**: Buah, Karbohidrat, Nasi, Protein, Sayur, Susu.
- **Backend**: Python 3.11 + Flask.
- **Library Utama**: OpenCV, NumPy, Ultralytics, Werkzeug.
- **Algoritma Estimasi**:
  `Berat (g) = Luas (cm²) × Tinggi Estimasi (cm) × Densitas Makanan (g/cm³)`

## 📁 Struktur Proyek
- `app.py`: Titik masuk utama API Flask.
- `core/`: Logika inti untuk deteksi, kalibrasi ompreng, dan estimasi berat.
- `utils/`: Helper untuk pemrosesan gambar dan pembentukan respons JSON.
- `training/`: Script dan panduan untuk fine-tuning model pada dataset custom.
- `models/`: Tempat penyimpanan file bobot model custom (`food_seg_model.pt`).
- `test_dashboard.html`: Halaman dashboard untuk testing API.

## 💻 Cara Menjalankan Secara Lokal
1. **Instalasi Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Menjalankan Server**:
   ```bash
   python app.py
   ```
3. **Akses Dashboard**:
   Buka file `test_dashboard.html` di browser Anda setelah server berjalan.

## 👤 Pengembang
**Tirtandro Meda**

---
*Proyek ini dikembangkan sebagai bagian dari fitur Smart Canteen untuk optimasi manajemen sampah makanan.*
