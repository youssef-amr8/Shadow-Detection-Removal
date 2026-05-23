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
        "--cv-method",
        default="lab-pro",
        choices=["lab-pro", "illumination", "hsv", "lab"],
        help="CV removal: lab-pro (default) or legacy methods",
    )
    p_run.add_argument(
        "--brightness",
        choices=["normal", "high", "max"],
        default="high",
        help=(
            "Shadow lift preset (lab-pro): normal | high (default) | max (strongest, matches lit background)"
        ),
    )
    p_run.add_argument(
        "--brightness-mode",
        choices=["match", "ratio"],
        default=None,
        help="Override preset: match=lift to lit mean (brightest), ratio=multiply L",
    )
    p_run.add_argument(
        "--brightness-boost",
        type=float,
        default=None,
        metavar="FACTOR",
        help="Extra gain on top of preset (e.g. 1.2). Tune in src/shadow_removal_cv.py if needed.",
    )
    p_run.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="Cap for ratio mode (default from preset; max preset uses 5.0)",
    )
    p_run.add_argument("--clahe", action="store_true", help="Enable CLAHE on L (can add grain; off by default)")
    p_run.add_argument(
        "--retinex",
        action="store_true",
        help="Optional Retinex on L before correction",
    )
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

        cv_kw = _lab_pro_kwargs(args) if args.cv_method == "lab-pro" else {}
        if args.cv_method == "lab-pro" and args.retinex:
            cv_kw["use_retinex"] = True

        run_on_image(
            args.image,
            args.output,
            detection_weights=args.detector_weights,
            save_mask_path=args.save_mask,
            cv_method=args.cv_method,
            cv_kwargs=cv_kw or None,
            verbose=args.verbose,
        )
        print(f"\nDone. Shadow-reduced image: {Path(args.output).resolve()}")
        if args.save_mask:
            print(f"      Detector mask:        {Path(args.save_mask).resolve()}")


def _lab_pro_kwargs(args: argparse.Namespace) -> dict:
    """Brightness presets for lab-pro (see also src/shadow_removal_cv.py defaults)."""
    presets = {
        "normal": {
            "brightness_mode": "ratio",
            "brightness_boost": 1.15,
            "max_ratio": 3.5,
            "min_ratio": 1.4,
            "match_chroma": True,
            "chroma_strength": 0.55,
        },
        "high": {
            "brightness_mode": "match",
            "brightness_boost": 1.1,
            "max_ratio": 4.0,
            "min_ratio": 1.5,
            "match_chroma": True,
            "chroma_strength": 0.65,
        },
        "max": {
            "brightness_mode": "match",
            "brightness_boost": 1.25,
            "max_ratio": 5.0,
            "min_ratio": 1.5,
            "match_chroma": True,
            "chroma_strength": 0.85,
            "mask_dilate_iter": 0,
            "blend_power": 1.0,
        },
    }
    kw = dict(presets.get(args.brightness, presets["high"]))
    if args.brightness_mode is not None:
        kw["brightness_mode"] = args.brightness_mode
    if args.brightness_boost is not None:
        kw["brightness_boost"] = args.brightness_boost
    if args.max_ratio is not None:
        kw["max_ratio"] = args.max_ratio
    if args.clahe:
        kw["use_clahe"] = True
    return kw


if __name__ == "__main__":
    main()
