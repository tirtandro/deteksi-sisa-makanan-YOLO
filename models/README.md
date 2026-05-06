# Models Directory

Simpan model YOLO yang sudah di-train di direktori ini.

## File yang Diharapkan

- `food_seg_model.pt` — Model YOLOv8 Segmentation yang sudah di-fine-tune untuk deteksi sisa makanan kantin.

## Cara Mendapatkan Model

### Opsi 1: Training Custom (Rekomendasi)
Lihat `training/README.md` untuk panduan lengkap training model.
Setelah training, model terbaik akan otomatis disalin ke sini.

### Opsi 2: Menggunakan Model Pre-trained (Quick Start)
Jika model custom belum tersedia, API akan otomatis menggunakan model pre-trained YOLOv8 (COCO).
Model ini bisa mendeteksi beberapa jenis makanan umum, tetapi akurasinya terbatas untuk sisa makanan kantin.

## Catatan
- File `.pt` **tidak** di-commit ke Git (terlalu besar)
- Tambahkan `*.pt` ke `.gitignore`
- Untuk deployment, sertakan model di Docker image atau download dari cloud storage saat startup
