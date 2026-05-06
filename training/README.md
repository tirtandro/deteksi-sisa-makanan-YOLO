# 🍽️ Panduan Training Model YOLO untuk Deteksi Sisa Makanan
## S3C-Smart-Canteen Food Waste Analysis

---

## 1. Persiapan Dataset

### 1.1 Mengumpulkan Data Gambar

Kumpulkan foto piring **setelah makan** dari kantin dengan variasi:

| Aspek | Variasi yang Dibutuhkan |
|:------|:------------------------|
| **Jenis makanan** | Nasi, mie, daging, ikan, sayuran, buah, roti, kuah |
| **Jumlah sisa** | Sedikit (~10%), sedang (~50%), banyak (~80%) |
| **Pencahayaan** | Terang, redup, cahaya alami, lampu neon |
| **Sudut kamera** | Atas (bird-eye), sedikit miring (~30°) |
| **Jenis piring** | Piring bulat, kotak, mangkuk, nampan |
| **Latar belakang** | Meja kayu, meja plastik, nampan |

**Rekomendasi**: Minimal **500-1000 gambar** untuk hasil yang baik. Lebih banyak = lebih akurat.

### 1.2 Memberi Anotasi (Labeling)

Gunakan salah satu tool berikut untuk membuat **polygon annotation** (segmentation mask):

1. **[Roboflow](https://roboflow.com/)** — Rekomendasi utama, ada fitur auto-augmentation
2. **[CVAT](https://cvat.ai/)** — Open source, powerful
3. **[Labelme](https://github.com/labelmeai/labelme)** — Desktop app, simple

#### Langkah Anotasi:
1. Upload gambar ke tool anotasi
2. Untuk setiap sisa makanan pada piring, buat **polygon** yang mengelilingi area makanan
3. Beri label sesuai kategori: `nasi`, `mie`, `daging`, `ikan`, `sayuran`, `buah`, `roti`, `kuah`, `lainnya`
4. Export dalam format **YOLO Segmentation**

### 1.3 Format Label YOLO Segmentation

Setiap gambar memiliki file `.txt` dengan format:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Contoh (`image001.txt`):
```
0 0.45 0.32 0.52 0.28 0.61 0.35 0.58 0.48 0.43 0.45
4 0.15 0.60 0.22 0.55 0.30 0.62 0.25 0.72 0.13 0.68
```

- Baris 1: Class 0 (nasi) dengan polygon coordinates (normalized 0-1)
- Baris 2: Class 4 (sayuran) dengan polygon coordinates

### 1.4 Struktur Folder Dataset

```
datasets/
└── food_waste/
    ├── train/
    │   ├── images/
    │   │   ├── img_001.jpg
    │   │   ├── img_002.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img_001.txt
    │       ├── img_002.txt
    │       └── ...
    ├── val/
    │   ├── images/
    │   │   └── ...
    │   └── labels/
    │       └── ...
    └── test/          # Optional
        ├── images/
        │   └── ...
        └── labels/
            └── ...
```

**Pembagian data yang disarankan**: 70% train / 20% val / 10% test

---

## 2. Konfigurasi Dataset

Edit file `training/data.yaml` sesuai lokasi dataset Anda:

```yaml
path: D:/Tirta/Project/AI berat makanan YOLO/datasets/food_waste
train: train/images
val: val/images
test: test/images

names:
  0: nasi
  1: mie
  2: daging
  3: ikan
  4: sayuran
  5: buah
  6: roti
  7: kuah
  8: lainnya

nc: 9
```

---

## 3. Training

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Jalankan Training

**Opsi 1: Menggunakan script (rekomendasi)**
```bash
# Training dasar (fastest, untuk prototyping)
python training/train.py --model yolov8n-seg.pt --epochs 100

# Training lebih akurat (medium model)
python training/train.py --model yolov8s-seg.pt --epochs 150 --batch 8

# Training terbaik (butuh GPU kuat)
python training/train.py --model yolov8m-seg.pt --epochs 200 --batch 4 --imgsz 640
```

**Opsi 2: Langsung via CLI ultralytics**
```bash
yolo task=segment mode=train \
    model=yolov8n-seg.pt \
    data=training/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16
```

**Opsi 3: Via Python (di notebook/script)**
```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
results = model.train(
    data='training/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
)
```

### 3.3 Pilihan Model

| Model | Size | Speed | Accuracy | GPU RAM | Rekomendasi |
|:------|:-----|:------|:---------|:--------|:------------|
| `yolov8n-seg.pt` | 6 MB | ⚡⚡⚡ | ★★☆ | ~4 GB | Prototyping, edge device |
| `yolov8s-seg.pt` | 23 MB | ⚡⚡ | ★★★ | ~6 GB | **Best balance** |
| `yolov8m-seg.pt` | 50 MB | ⚡ | ★★★★ | ~8 GB | Produksi (GPU available) |
| `yolov8l-seg.pt` | 83 MB | 🐌 | ★★★★★ | ~12 GB | Highest accuracy |

---

## 4. Evaluasi Model

Setelah training selesai, evaluasi model pada validation set:

```bash
yolo task=segment mode=val \
    model=runs/segment/food_waste_seg/weights/best.pt \
    data=training/data.yaml
```

### Metrik yang Perlu Diperhatikan:
- **mAP50**: Mean Average Precision @ IoU 0.5 → target > 0.7
- **mAP50-95**: Mean Average Precision @ IoU 0.5-0.95 → target > 0.5
- **Per-class AP**: Pastikan semua class memiliki AP yang baik

---

## 5. Deploy Model ke API

Setelah training, model terbaik otomatis disalin ke `models/food_seg_model.pt`.
Jika menggunakan script manual:

```bash
# Copy model terbaik
cp runs/segment/food_waste_seg/weights/best.pt models/food_seg_model.pt

# Jalankan API
python app.py
```

---

## 6. Tips untuk Akurasi Optimal

1. **Data Quality > Data Quantity**: 500 gambar berkualitas > 2000 gambar buruk
2. **Konsistensi Anotasi**: Pastikan semua annotator menggunakan standar yang sama
3. **Augmentasi Data**: Script training sudah termasuk augmentation (flip, scale, HSV, mosaic)
4. **Iterative Improvement**: Setelah deploy, kumpulkan gambar yang gagal terdeteksi → tambahkan ke dataset → re-train
5. **Kalibrasi Berat**: Bandingkan estimasi berat dengan timbangan fisik → sesuaikan konstanta densitas/tinggi di `core/food_constants.py`

---

## 7. Quick Start dengan Roboflow

Jika ingin cara tercepat:

1. Buat akun di [roboflow.com](https://roboflow.com)
2. Create new project → Instance Segmentation
3. Upload gambar → Annotate dengan polygon tool
4. Generate dataset → Export dalam format "YOLOv8"
5. Download dan extract ke folder `datasets/food_waste/`
6. Jalankan: `python training/train.py`
