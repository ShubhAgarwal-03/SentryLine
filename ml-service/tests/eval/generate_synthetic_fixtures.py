"""Generates placeholder images so the eval harness can be smoke-tested
without real photos. Run once, then `run_eval.py --manifest synthetic_manifest.json`.

Reminder (see README.md): these images are crude synthetic shapes. YOLO and
CLIP were never trained on them, so scores here test "does the harness
code run without crashing," not "is the model accurate." Do not report
these numbers as real accuracy.
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).parent / "fixtures" / "_synthetic_smoketest"


def _make_image(path: Path, box_color: str, box: tuple[int, int, int, int]):
    img = Image.new("RGB", (640, 480), color="gray")
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, fill=box_color, outline="black", width=3)
    img.save(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _make_image(OUTPUT_DIR / "smoketest_1.jpg", "tan", (150, 80, 320, 420))
    _make_image(OUTPUT_DIR / "smoketest_2.jpg", "silver", (250, 150, 420, 260))

    print(f"Wrote synthetic smoke-test images to {OUTPUT_DIR}")
    print("Now run: python -m tests.eval.run_eval --manifest tests/eval/synthetic_manifest.json")


if __name__ == "__main__":
    main()