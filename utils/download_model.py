import os
from roboflow import Roboflow

# ──────────────────────────────────────────────
# Konfigurasi Roboflow
# ──────────────────────────────────────────────
API_KEY = "8SEqM5jS0qJ4PfUjzQ1h"  # API Key Anda dari screenshot
WORKSPACE_ID = "new-workspace"
PROJECT_ID = "foodseg103-v1-gcgrs"
VERSION_NUMBER = 1

# Folder tujuan
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
TARGET_FILENAME = "food_seg_model.pt"

def download_trained_model():
    print(f"Memulai download model dari Roboflow...")
    print(f"Project: {PROJECT_ID} (Version {VERSION_NUMBER})")
    
    try:
        # Inisialisasi Roboflow
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        version = project.version(VERSION_NUMBER)
        
        # Buat folder models jika belum ada
        os.makedirs(TARGET_DIR, exist_ok=True)
        
        # Download model weights (Format YOLOv8/Ultralytics)
        # Note: Ini akan mendownload zip berisi best.pt
        print("Sedang mengambil link download...")
        dataset = version.download("yolov8")
        
        # Cari file best.pt di dalam folder download
        # Roboflow biasanya mendownload ke folder lokal
        source_path = os.path.join(dataset.location, "weights", "best.pt")
        
        if not os.path.exists(source_path):
            # Coba cari di root folder download jika tidak ada folder weights
            source_path = os.path.join(dataset.location, "best.pt")

        if os.path.exists(source_path):
            target_path = os.path.join(TARGET_DIR, TARGET_FILENAME)
            import shutil
            shutil.copy2(source_path, target_path)
            print(f"\nBERHASIL!")
            print(f"Model disimpan di: {target_path}")
            print(f"Anda sekarang bisa menjalankan 'python app.py'")
        else:
            print("\nError: File best.pt tidak ditemukan di folder download.")
            print(f"Cek isi folder: {dataset.location}")

    except Exception as e:
        print(f"\nGAGAL: {e}")
        print("Pastikan nomor versi (VERSION_NUMBER) sudah memiliki model yang terlatih di Roboflow.")

if __name__ == "__main__":
    download_trained_model()
