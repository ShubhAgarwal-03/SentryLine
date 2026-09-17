"""Fine-tunes YOLOv8n on the hazard class taxonomy, starting from the stock
COCO-pretrained checkpoint (transfer learning, not training from scratch —
the stock model already knows general object shapes; we're teaching it a
handful of new classes on top of that).

Usage:
    python train.py --data data.yaml --epochs 100 --batch 16

Run prepare_dataset.py first to generate data.yaml.
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml from prepare_dataset.py")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Starting checkpoint (must match app/perception/detector.py's MODEL_PATH if you intend to deploy the result)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs", help="Output directory for training runs")
    parser.add_argument("--name", default="sentryline_hazard_v1", help="Run name")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker processes")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (can cause NaN losses on some GPU/data combos — off by default)")
    args = parser.parse_args()

    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        amp=args.amp,
        project=args.project,
        name=args.name,
        # Modest augmentation defaults are fine to start — small validation
        # sets don't benefit much from aggressive augmentation search;
        # revisit once there's enough data to tell signal from noise.
    )

    print(f"\nTraining complete. Best weights: {args.project}/{args.name}/weights/best.pt")
    print(
        "To deploy: copy best.pt to apps/ml-service/models/ and update "
        "MODEL_PATH in app/perception/detector.py to point at it. Then run "
        "the eval harness (tests/eval/eval_detection.py) against real "
        "fixtures to confirm mAP actually improved before replacing the "
        "production checkpoint."
    )


if __name__ == "__main__":
    main()