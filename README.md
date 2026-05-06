# S3C-Smart-Canteen: Sistem Analisis Sisa Makanan

Sistem analisis sisa makanan berbasis Artificial Intelligence (AI) untuk aplikasi **S3C-Smart-Canteen**. Proyek ini menggunakan model YOLO (You Only Look Once) untuk *instance segmentation* guna mendeteksi jenis sisa makanan dan mengestimasi beratnya (dalam gram) berdasarkan foto 2D piring setelah makan.

## 🚀 Fitur Utama
- **Deteksi Sisa Makanan**: Menggunakan YOLOv8-seg untuk identifikasi otomatis berbagai kategori makanan (nasi, daging, sayuran, dll).
- **Estimasi Berat Akurat**: Algoritma matematis yang mengonversi luas piksel ke skala dunia nyata menggunakan kalibrasi objek referensi (piring standar).
- **API Backend**: Dibangun dengan Flask, siap diintegrasikan dengan frontend atau aplikasi mobile.
- **Kalibrasi Otomatis**: Mendeteksi piring dalam gambar untuk menentukan skala cm/pixel secara dinamis.
- **Siap Deployment**: Konfigurasi lengkap untuk deployment ke Google Cloud Platform (GCP) menggunakan Cloud Run atau App Engine.

## 🛠️ Spesifikasi Teknis
- **Model AI**: YOLOv8 Segmentation (Ultralytics).
- **Backend**: Python 3.11 + Flask.
- **Library Utama**: OpenCV, NumPy, Pytest, Gunicorn.
- **Algoritma Estimasi**:
  `Berat (g) = Luas (cm²) × Tinggi Estimasi (cm) × Densitas Makanan (g/cm³)`

## 📁 Struktur Proyek
- `app.py`: Titik masuk utama API Flask.
- `core/`: Logika inti untuk deteksi, kalibrasi piring, dan estimasi berat.
- `utils/`: Helper untuk pemrosesan gambar dan pembentukan respons JSON.
- `training/`: Script dan panduan untuk fine-tuning model pada dataset custom.
- `models/`: Tempat penyimpanan file bobot model (.pt).
- `tests/`: Suite pengujian otomatis untuk memastikan keandalan API.

## 💻 Cara Menjalankan Secara Lokal
1. **Instalasi Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Menjalankan Server**:
   ```bash
   python app.py
   ```
3. **Akses API**:
   Server akan berjalan di `http://localhost:8080`. Gunakan endpoint `/api/analyze` dengan metode POST untuk mengirim gambar.

## 🧪 Pengujian
Jalankan pengujian otomatis dengan perintah:
```bash
pytest tests/test_api.py -v
```

## 👤 Pengembang
**Tirtandro Meda**

---
*Proyek ini dikembangkan sebagai bagian dari fitur Smart Canteen untuk optimasi manajemen sampah makanan.*
