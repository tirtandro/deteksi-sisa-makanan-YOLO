"""
S3C-Smart-Canteen: Food Constants
Tabel konstanta densitas dan tinggi estimasi untuk berbagai jenis makanan.
Digunakan untuk mengkonversi volume (cm³) ke berat (gram).

Sumber referensi densitas:
- ResearchGate: Food Density Database
- USDA Food Composition Databases
- FAO Technical Papers

Note: Nilai-nilai ini adalah aproksimasi untuk makanan yang sudah dimasak (cooked).
Akurasi dapat ditingkatkan dengan kalibrasi menggunakan data pengukuran real.
"""


# ──────────────────────────────────────────────
# Food Class Definitions
# ──────────────────────────────────────────────
# Mapping dari class ID model YOLO ke nama kategori
# HARUS sesuai dengan model.names dari model custom yang di-train!
# Model saat ini: {0: 'buah', 1: 'karbo', 2: 'nasi', 3: 'protein', 4: 'sayur', 5: 'susu'}
FOOD_CLASSES = {
    0: "buah",
    1: "karbo",
    2: "nasi",
    3: "protein",
    4: "sayur",
    5: "susu",
}

# Mapping nama class ke label yang user-friendly (Bahasa Indonesia)
FOOD_LABELS = {
    "buah": "Buah / Fruit",
    "karbo": "Karbohidrat / Carbs",
    "nasi": "Nasi / Rice",
    "protein": "Protein",
    "sayur": "Sayur / Vegetables",
    "susu": "Susu / Milk",
    # Legacy labels (for COCO fallback compatibility)
    "lainnya": "Lainnya / Others",
}

# ──────────────────────────────────────────────
# Food Density Table (g/cm³)
# ──────────────────────────────────────────────
# Bulk density untuk makanan yang sudah dimasak.
# Bulk density memperhitungkan udara antar potongan makanan.
FOOD_DENSITY = {
    "buah": 0.65,       # Buah potong (semangka, apel, jeruk, dll.)
    "karbo": 0.55,      # Karbohidrat umum (mie, roti, kentang, dll.)
    "nasi": 0.75,       # Nasi putih matang, loosely packed
    "protein": 1.00,    # Protein (daging ayam, ikan, telur, tempe, dll.)
    "sayur": 0.45,      # Sayuran matang, mixed/chopped
    "susu": 1.03,       # Susu / minuman (liquid density ~1.03 g/cm³)
    # Legacy (for COCO fallback)
    "lainnya": 0.70,    # Default untuk makanan tidak terklasifikasi
}

# ──────────────────────────────────────────────
# Food Estimated Height Table (cm)
# ──────────────────────────────────────────────
# Estimasi ketebalan/tinggi tumpukan makanan pada piring.
# Digunakan untuk mengkonversi area 2D ke volume 3D.
# Asumsi: makanan tersebar merata di area mask.
FOOD_HEIGHT = {
    "buah": 1.5,        # Buah potong
    "karbo": 2.0,       # Karbohidrat (mie, roti) cenderung tebal
    "nasi": 1.5,        # Nasi biasanya bertumpuk ~1.5 cm
    "protein": 1.5,     # Potongan daging/lauk rata-rata tebal 1.5 cm
    "sayur": 2.0,       # Sayuran bervariasi, cenderung voluminous
    "susu": 5.0,        # Susu kotak / cup (tinggi kemasan)
    # Legacy (for COCO fallback)
    "lainnya": 1.5,     # Default
}

# ──────────────────────────────────────────────
# Height Adjustment Factor
# ──────────────────────────────────────────────
# Faktor penyesuaian untuk mengoreksi estimasi tinggi berdasarkan
# rasio luas mask terhadap luas piring.
# Jika sisa makanan hanya sedikit (kecil), kemungkinan lebih tipis.
# Jika banyak, kemungkinan lebih tebal/bertumpuk.
def get_height_adjustment(mask_area_ratio: float) -> float:
    """
    Menghitung faktor penyesuaian tinggi berdasarkan rasio
    luas mask terhadap luas piring.

    Args:
        mask_area_ratio: Rasio luas mask / luas piring (0.0 - 1.0)

    Returns:
        Adjustment factor (0.5 - 1.5)
    """
    if mask_area_ratio < 0.05:
        return 0.5   # Sedikit sekali → sangat tipis
    elif mask_area_ratio < 0.15:
        return 0.7   # Sedikit → tipis
    elif mask_area_ratio < 0.30:
        return 1.0   # Sedang → normal
    elif mask_area_ratio < 0.50:
        return 1.2   # Banyak → sedikit tebal
    else:
        return 1.5   # Sangat banyak → bertumpuk


# ──────────────────────────────────────────────
# COCO Class Mapping (for pre-trained model fallback)
# ──────────────────────────────────────────────
# Ketika menggunakan model pre-trained COCO, beberapa class yang
# relevan dipetakan ke kategori makanan kita.
COCO_TO_FOOD_MAPPING = {
    # COCO class_id: our food category
    46: "lainnya",    # banana → buah
    47: "lainnya",    # apple → buah
    48: "lainnya",    # sandwich → roti
    49: "lainnya",    # orange → buah
    50: "lainnya",    # broccoli → sayuran
    51: "lainnya",    # carrot → sayuran
    52: "lainnya",    # hot dog → daging
    53: "lainnya",    # pizza → lainnya
    54: "lainnya",    # donut → roti
    55: "lainnya",    # cake → roti
}

# Semua COCO food-related class IDs
COCO_FOOD_CLASS_IDS = set(COCO_TO_FOOD_MAPPING.keys())
