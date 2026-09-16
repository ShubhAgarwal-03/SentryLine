"""Detection accuracy — PRD §15 'Perception layer': mAP@0.5.

Implements a simplified single-IoU-threshold average precision, matched
per-class. This is not a byte-for-byte reimplementation of COCO's official
mAP (no interpolation across the full 0.5:0.95 sweep) — it's built to be
readable and auditable over being bit-exact with pycocotools, which matters
more for a small hand-curated validation set than for leaderboard-grade
benchmarking.
"""
import argparse
from collections import defaultdict
from pathlib import Path

from app.perception.detector import detect_objects

from .common import ManifestEntry, iou, load_image_bgr, load_manifest

IOU_THRESHOLD = 0.5


def _average_precision(matches: list[tuple[float, bool]]) -> float:
    """matches: list of (confidence, is_true_positive), for one class.
    Returns AP via the standard precision-recall area calculation.
    """
    if not matches:
        return 0.0

    matches.sort(key=lambda m: m[0], reverse=True)
    total_positives = sum(1 for _, is_tp in matches if is_tp)
    if total_positives == 0:
        return 0.0

    tp_cumulative, fp_cumulative = 0, 0
    precisions, recalls = [], []
    for _, is_tp in matches:
        if is_tp:
            tp_cumulative += 1
        else:
            fp_cumulative += 1
        precisions.append(tp_cumulative / (tp_cumulative + fp_cumulative))
        recalls.append(tp_cumulative / total_positives)

    # Area under the precision-recall curve (simple trapezoidal, not the
    # 11-point interpolation COCO uses — documented simplification, see
    # module docstring).
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap


def evaluate_detection(entries: list[ManifestEntry]) -> dict:
    # per_class[label] = list of (confidence, is_true_positive)
    per_class: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    total_gt_boxes = 0

    for entry in entries:
        image = load_image_bgr(entry.image_path)
        predictions = detect_objects(image)
        gt_remaining = list(entry.expected_objects)
        total_gt_boxes += len(gt_remaining)

        # Greedy match: for each prediction (highest confidence first),
        # claim the best unclaimed ground-truth box of the same class if
        # IoU clears the threshold.
        for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
            best_match_idx, best_iou = None, 0.0
            for idx, gt in enumerate(gt_remaining):
                if gt.label != pred.label:
                    continue
                score = iou(pred.bbox, gt.bbox)
                if score > best_iou:
                    best_iou, best_match_idx = score, idx

            is_tp = best_iou >= IOU_THRESHOLD
            per_class[pred.label].append((pred.confidence, is_tp))
            if is_tp:
                gt_remaining.pop(best_match_idx)

    ap_per_class = {label: _average_precision(matches) for label, matches in per_class.items()}
    mean_ap = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0

    return {
        "mAP@0.5": mean_ap,
        "ap_per_class": ap_per_class,
        "num_images": len(entries),
        "num_ground_truth_boxes": total_gt_boxes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parent / "manifest.json"),
        help="Path to manifest.json (default: real fixtures manifest)",
    )
    args = parser.parse_args()

    entries = load_manifest(Path(args.manifest))
    if not entries:
        print(
            "No manifest entries found. This is expected on a fresh checkout — "
            "see tests/eval/README.md to add real labeled photos before this "
            "script produces a meaningful number."
        )
        return

    results = evaluate_detection(entries)
    print(f"mAP@0.5: {results['mAP@0.5']:.3f}  "
          f"({results['num_images']} images, {results['num_ground_truth_boxes']} ground-truth boxes)")
    for label, ap in sorted(results["ap_per_class"].items()):
        print(f"  {label:<15} AP={ap:.3f}")


if __name__ == "__main__":
    main()