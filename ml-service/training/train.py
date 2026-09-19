"""Fine-tunes YOLOv8n on the hazard class taxonomy, starting from the stock
COCO-pretrained checkpoint (transfer learning, not training from scratch —
the stock model already knows general object shapes; we're teaching it a
handful of new classes on top of that).

Usage:
    python train.py --data data.yaml --epochs 100 --batch 16

Run prepare_dataset.py first to generate data.yaml.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml from prepare_dataset.py")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Starting checkpoint for fine-tuning (the stock COCO model)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs", help="Output directory for training runs")
    parser.add_argument("--name", default="sentryline_hazard_v1", help="Run name")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker processes")
    parser.add_argument("--patience", type=int, default=30, help="Stop early after this many epochs with no val improvement (Ultralytics default is 100, i.e. never for a 100-epoch run)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (can cause NaN losses on some GPU/data combos — off by default)")
    args = parser.parse_args()

    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        amp=args.amp,
        project=args.project,
        name=args.name,
        # Modest augmentation defaults are fine to start — small validation
        # sets don't benefit much from aggressive augmentation search;
        # revisit once there's enough data to tell signal from noise.
    )

    # Ultralytics auto-increments the run folder (name -> name2, name3, ...)
    # when exist_ok=False, so args.name is NOT the real path. Ask the trainer.
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print(
        f"To deploy: copy {best} to ml-service/models/gun_fire_v1.pt (or set "
        "SENTRYLINE_HAZARD_MODEL). app/perception/detector.py runs this alongside "
        "the stock yolov8n.pt — do NOT replace yolov8n.pt with it, or person/knife/"
        "etc. stop being detected. Then run tests/eval/eval_detection.py against "
        "real fixtures to confirm mAP actually improved."
    )


if __name__ == "__main__":
    main()