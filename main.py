"""Shadow detection (train) + detect-and-remove (CV) CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train shadow detection, or run detection + CV shadow removal on an image."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_det = sub.add_parser("train-detection", help="Train shadow-mask U-Net → models/shadow_detection.pth")
    p_det.add_argument("--data", default="data/ISTD_Dataset", help="ISTD root with train/train_A and train/train_B")
    p_det.add_argument("--epochs", type=int, default=50)
    p_det.add_argument("--batch-size", type=int, default=4)
    p_det.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device to use for training (auto by default)")

    p_run = sub.add_parser(
        "run",
        help="Load image → detector mask → CV (LAB) removal → save result (and optional mask PNG)",
    )
    p_run.add_argument("--image", required=True, help="Input image path")
    p_run.add_argument("--output", default="outputs/shadow_removed.png", help="Output image after CV removal")
    p_run.add_argument(
        "--save-mask",
        default=None,
        metavar="PATH",
        help="If set, save the predicted shadow mask as a grayscale PNG (e.g. outputs/mask.png)",
    )
    p_run.add_argument("--detector-weights", default=None, help="Default: models/shadow_detection.pth")
    p_run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each pipeline step (weights → mask → CV removal → output file)",
    )

    args = parser.parse_args()

    if args.command == "train-detection":
        from shadow_detection import train_shadow_detection

        train_shadow_detection(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "run":
        from pipeline import run_on_image

        run_on_image(
            args.image,
            args.output,
            detection_weights=args.detector_weights,
            save_mask_path=args.save_mask,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
