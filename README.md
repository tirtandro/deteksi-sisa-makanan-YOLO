# S3C-Smart-Canteen: Sistem Analisis Sisa Makanan (YOLO Backend)

Sistem analisis sisa makanan berbasis Artificial Intelligence (AI) untuk aplikasi **S3C-Smart-Canteen**. Proyek ini menggunakan model YOLO11 untuk *instance segmentation* guna mendeteksi jenis sisa makanan dan mengestimasi beratnya (dalam gram).

## 🚀 Fitur Utama
- **Deteksi Sisa Makanan**: Menggunakan YOLO11 Segmentation untuk identifikasi otomatis kategori makanan.
- **Spatial Filtering**: Fitur cerdas untuk mengabaikan deteksi objek yang berada di luar area piring (mengurangi *false positive* dari latar belakang/taplak meja).
- **Estimasi Berat Akurat**: Algoritma matematis yang mengonversi luas piksel ke skala dunia nyata menggunakan kalibrasi radius piring standar.
- **High Reliability**: Tetap menghasilkan gambar anotasi (foto dashboard) meskipun sisa makanan terdeteksi 0 gram, memastikan konsistensi tampilan pada UI.
- **Optimized Threshold**: Menggunakan Confidence Threshold 0.45 untuk keseimbangan terbaik antara sensitivitas dan presisi.

## 🛠️ Spesifikasi Teknis
- **Model AI**: YOLO11 Instance Segmentation (Ultralytics).
- **Kategori Terdeteksi**: Buah, Karbohidrat, Nasi, Protein, Sayur, Susu, Lainnya.
- **Backend**: Python 3.11 + Flask.
- **Library Utama**: OpenCV, NumPy, Ultralytics, Werkzeug.
- **Algoritma Estimasi**:
  `Berat (g) = Luas (cm²) × Tinggi Estimasi (cm) × Densitas Makanan (g/cm³)`

## 📁 Struktur Proyek
- `app.py`: Titik masuk utama API Flask (Endpoint: `/api/analyze`).
- `core/`: 
  - `detector.py`: Logika inferensi YOLO11.
  - `plate_calibrator.py`: Deteksi piring & kalkulasi skala cm/pixel.
  - `estimator.py`: Kalkulasi berat berdasarkan luas mask.
- `utils/`: 
  - `image_utils.py`: Pemrosesan gambar, resizing, dan rendering anotasi.
  - `response_builder.py`: Formatter respons JSON standar.
- `models/`: Penyimpanan file bobot model (`food_seg_model.pt`).

## 👤 Pengembang
**Tirtandro Meda**

---
*Proyek ini dikembangkan sebagai bagian dari fitur Smart Canteen untuk optimasi manajemen sampah makanan.*
