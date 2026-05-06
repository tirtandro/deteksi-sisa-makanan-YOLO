"""
S3C-Smart-Canteen: YOLO Training Script
========================================
Script untuk fine-tuning model YOLOv8 Segmentation pada dataset
sisa makanan kantin.

Usage:
    # Training dengan default config
    python training/train.py

    # Training dengan custom parameters
    python training/train.py --model yolov8s-seg.pt --epochs 200 --batch 8 --imgsz 640

    # Resume training
    python training/train.py --resume
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 Segmentation model for food waste detection"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Pre-trained model to fine-tune. Options: yolov8n-seg.pt, "
             "yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt "
             "(default: yolov8n-seg.pt — fastest, good for starting)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "training" / "data.yaml"),
        help="Path to dataset YAML config (default: training/data.yaml)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Use -1 for auto-batch (default: 16)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on: 'cpu', '0', '0,1', etc. "
             "Empty string for auto-detect (default: auto)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="food_waste_seg",
        help="Experiment name for saving results (default: food_waste_seg)",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=str(project_root / "runs" / "segment"),
        help="Project directory for saving results",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export best model to ONNX after training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("S3C-Smart-Canteen: YOLO Food Waste Segmentation Training")
    print("=" * 60)

    # Import YOLO
    from ultralytics import YOLO

    # ── Validate dataset ──
    if not os.path.exists(args.data):
        print(f"\n❌ ERROR: Dataset config not found: {args.data}")
        print("\nPlease create your dataset first. See training/README.md for guide.")
        sys.exit(1)

    # ── Load model ──
    if args.resume:
        # Resume from last checkpoint
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"\n❌ ERROR: No checkpoint found at: {last_pt}")
            sys.exit(1)
        print(f"\n📦 Resuming training from: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"\n📦 Loading pre-trained model: {args.model}")
        model = YOLO(args.model)

    # ── Training Configuration ──
    train_config = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "name": args.name,
        "project": args.project,
        "workers": args.workers,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "save": True,
        "save_period": 10,  # Save checkpoint setiap 10 epoch
        "plots": True,

        # Data Augmentation (optimized for food images)
        "hsv_h": 0.015,       # Hue augmentation
        "hsv_s": 0.7,         # Saturation augmentation
        "hsv_v": 0.4,         # Value/brightness augmentation
        "degrees": 10.0,      # Rotation (±10°)
        "translate": 0.1,     # Translation
        "scale": 0.5,         # Scale
        "flipud": 0.0,        # No vertical flip (food orientation matters)
        "fliplr": 0.5,        # Horizontal flip
        "mosaic": 1.0,        # Mosaic augmentation
        "mixup": 0.1,         # MixUp augmentation
    }

    # Add device if specified
    if args.device:
        train_config["device"] = args.device

    print("\n📋 Training Configuration:")
    for key, value in train_config.items():
        print(f"   {key}: {value}")

    # ── Start Training ──
    print("\n🚀 Starting training...\n")
    results = model.train(**train_config)

    # ── Post-Training ──
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)

    # Best model path
    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n📁 Best model saved to: {best_model}")

    # Validate on validation set
    print("\n📊 Running validation...")
    val_results = model.val()
    print(f"   mAP50: {val_results.seg.map50:.4f}")
    print(f"   mAP50-95: {val_results.seg.map:.4f}")

    # ── Copy best model to models/ directory ──
    target_model_path = project_root / "models" / "food_seg_model.pt"
    os.makedirs(target_model_path.parent, exist_ok=True)

    import shutil
    shutil.copy2(str(best_model), str(target_model_path))
    print(f"\n📦 Model copied to: {target_model_path}")
    print("   This model will be automatically loaded by the Flask API.")

    # ── Export to ONNX (optional) ──
    if args.export:
        print("\n📤 Exporting model to ONNX...")
        model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print("   ONNX export complete!")

    print("\n🎉 All done! You can now run the Flask API with the trained model:")
    print("   python app.py")


if __name__ == "__main__":
    main()
