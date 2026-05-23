"""Shadow detection training + learned removal CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_learned_removal(
    image_path: str | Path,
    output_path: str | Path,
    *,
    detector_weights: str | Path | None = None,
    removal_weights: str | Path | None = None,
    device: str | None = None,
    save_mask_path: str | Path | None = None,
    verbose: bool = False,
) -> None:
    from shadow_detector import ShadowDetector
    from shadow_removal_net import ShadowRemovalNet

    detector = ShadowDetector(detector_weights, device=device)
    remover = ShadowRemovalNet(removal_weights, device=device)

    if verbose:
        print(f"[1/3] Loaded detector from {detector.weights_path}")
        print(f"[2/3] Loaded removal net from {remover.weights_path}")
        print(f"[3/3] Predicting mask and removing shadows for {Path(image_path).resolve()}")

    rgb, mask = detector.predict_mask_from_path(image_path)
    out_rgb = remover.predict(rgb, mask)

    if save_mask_path is not None:
        mp = Path(save_mask_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        m = mask if mask.ndim == 2 else mask[:, :, 0]
        cv2.imwrite(str(mp), (m.clip(0, 1) * 255.0).astype(np.uint8))
        if verbose:
            print(f"      Saved mask: {mp.resolve()}")

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_bgr = cv2.cvtColor((out_rgb * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(outp), out_bgr)
    if verbose:
        print(f"      Saved output: {outp.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train shadow detection or run learned detector+removal on one image."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_det = sub.add_parser("train-detection", help="Train shadow-mask U-Net → models/shadow_detection.pth")
    p_det.add_argument("--data", default="data/ISTD_Dataset", help="ISTD root with train/train_A and train/train_B")
    p_det.add_argument("--epochs", type=int, default=50)
    p_det.add_argument("--batch-size", type=int, default=4)
    p_det.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device to use for training (auto by default)")

    p_rem = sub.add_parser(
        "train-removal",
        help="Train removal U-Net (train_A+train_B→train_C) → models/shadow_removal.pth",
    )
    p_rem.add_argument(
        "--data",
        default="data/ISTD_Dataset",
        help="ISTD root with train_A, train_B, train_C",
    )
    p_rem.add_argument("--epochs", type=int, default=50)
    p_rem.add_argument("--batch-size", type=int, default=8)
    p_rem.add_argument("--img-size", type=int, default=256)
    p_rem.add_argument("--lr", type=float, default=1e-4)
    p_rem.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 0 on Colab)")
    p_rem.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p_rem.add_argument("--save", default=None, help="Output .pth path (default: models/shadow_removal.pth)")

    p_run = sub.add_parser(
        "run",
        help="Run detection + learned removal on one image.",
    )
    p_run.add_argument(
        "--image",
        required=True,
        help="Input image path",
    )
    p_run.add_argument(
        "--output",
        default="outputs/shadow_removed.png",
        help="Output image after learned shadow removal",
    )
    p_run.add_argument(
        "--save-mask",
        default=None,
        metavar="PATH",
        help="Save the predicted shadow mask as a grayscale PNG",
    )
    p_run.add_argument(
        "--detector-weights",
        default=None,
        help="Detector weights .pth (default: models/shadow_detection.pth)",
    )
    p_run.add_argument(
        "--removal-weights",
        default=None,
        help="Removal weights .pth (default: models/shadow_removal.pth)",
    )
    p_run.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for inference (auto by default)",
    )
    p_run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print pipeline steps and save paths",
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
    elif args.command == "train-removal":
        from shadow_removal_train import train_shadow_removal

        train_shadow_removal(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            device=args.device,
            num_workers=args.num_workers,
            save_path=args.save,
        )
    elif args.command == "run":
        run_learned_removal(
            args.image,
            args.output,
            detector_weights=args.detector_weights,
            removal_weights=args.removal_weights,
            device=args.device,
            save_mask_path=args.save_mask,
            verbose=args.verbose,
        )
        print(f"\nDone. Shadow-reduced image: {Path(args.output).resolve()}")
        if args.save_mask:
            print(f"      Detector mask:        {Path(args.save_mask).resolve()}")


if __name__ == "__main__":
    main()
