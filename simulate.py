import requests
import json
import base64
import os
import cv2
import numpy as np

API_URL = "http://127.0.0.1:8080/api/analyze"
IMAGE_PATH = "sample_plate.jpg"
OUTPUT_PATH = "result_annotated.jpg"

def create_sample_image():
    """Membuat gambar sintetis piring dengan makanan (nasi & sayuran) untuk testing."""
    print("Membuat gambar sampel (sample_plate.jpg)...")
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Latar belakang gelap

    # Gambar piring (lingkaran putih)
    cv2.circle(img, (320, 320), 250, (220, 220, 220), -1)

    # Gambar "Nasi" (warna putih kekuningan) - COCO class sandwich/cake yang akan dimap ke roti/lainnya 
    # karena kita pakai pre-trained model saat ini. Mari kita gambar bentuk yang mungkin dideteksi.
    # Karena kita pakai pre-trained COCO, kita gambar apel dan pisang saja (class 47 dan 46)
    # yang di mapping kita akan masuk ke "buah".
    
    # Gambar sesuatu menyerupai apel (merah)
    cv2.circle(img, (250, 280), 60, (50, 50, 220), -1) 
    
    # Gambar sesuatu menyerupai brokoli (hijau gelap) - class 50 di COCO
    cv2.circle(img, (400, 320), 50, (30, 120, 30), -1)
    cv2.circle(img, (430, 290), 45, (30, 120, 30), -1)
    cv2.circle(img, (380, 280), 40, (30, 120, 30), -1)

    cv2.imwrite(IMAGE_PATH, img)
    print("Gambar sampel berhasil dibuat!\n")

def simulate_upload():
    print(f"Mengirim {IMAGE_PATH} ke API {API_URL} ...")
    
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/jpeg")}
        data = {"include_annotated": "true"}
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            
            print("\n" + "="*50)
            print("HASIL ANALISIS API")
            print("="*50)
            
            if result.get("success"):
                summary = result["summary"]
                print(f"Status: Berhasil")
                print(f"Pesan: {result['message']}")
                print(f"Waktu Proses: {result['processing_time_ms']} ms")
                print(f"Jumlah Item Terdeteksi: {summary['total_items_detected']}")
                print(f"Total Estimasi Berat: {summary['total_waste_weight_grams']} gram")
                
                print("\nDetail Deteksi:")
                for i, det in enumerate(result["detections"], 1):
                    print(f"  {i}. {det['food_label']} | Confidence: {det['confidence']:.2f} | Estimasi Berat: {det['estimated_weight_grams']}g")
                
                # Simpan gambar beranotasi jika ada
                if "annotated_image_base64" in result:
                    img_data = base64.b64decode(result["annotated_image_base64"])
                    with open(OUTPUT_PATH, "wb") as out_file:
                        out_file.write(img_data)
                    print(f"\n[INFO] Gambar hasil deteksi (beranotasi) disimpan ke: {OUTPUT_PATH}")
                    
            else:
                print("API mengembalikan status gagal:")
                print(json.dumps(result, indent=2))
                
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Gagal menghubungi API: {e}")
            print("Pastikan server Flask sudah berjalan (python app.py) di terminal lain.")

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        create_sample_image()
    simulate_upload()
