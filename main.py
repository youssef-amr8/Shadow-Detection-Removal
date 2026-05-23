"""Shadow detection training + learned removal CLI."""

from __future__ import annotations

import argparse
import json
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
        print(f"[1/3] Loaded detector from {detector_weights or 'models/shadow_detection.pth'}")
        print(f"[2/3] Loaded removal net from {removal_weights or 'models/shadow_removal.pth'}")
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


def run_inference_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    mask_dir: str | Path | None = None,
    *,
    detector_weights: str | Path | None = None,
    removal_weights: str | Path | None = None,
    device: str | None = None,
    num_images: int = 20,
    seed: int | None = None,
    gt_dir: str | Path | None = None,
    verbose: bool = False,
) -> None:
    """Run shadow detection and removal on random images and compute metrics."""
    import torch

    from shadow_detector import ShadowDetector
    from shadow_removal_net import ShadowRemovalNet
    from metrics import MetricsComputer

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
    if gt_dir is not None:
        gt_dir = Path(gt_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[1/3] Loading shadow detector from {detector_weights}")
    detector = ShadowDetector(detector_weights, device=device)

    if verbose:
        print(f"[2/3] Loading shadow remover from {removal_weights}")
    remover = ShadowRemovalNet(removal_weights, device=device)

    # Get image list and sample randomly
    image_files = [f for f in input_dir.glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")

    image_files = sorted(image_files)
    if len(image_files) > num_images:
        sampled_indices = np.random.choice(len(image_files), num_images, replace=False)
        image_files = [image_files[i] for i in sorted(sampled_indices)]

    if verbose:
        print(f"[3/3] Processing {len(image_files)} random images")

    # Inference loop
    processed_images = []
    metrics_list = []

    for img_path in image_files:
        rgb, mask = detector.predict_mask_from_path(img_path)
        out_rgb = remover.predict(rgb, mask)

        # Save output
        out_bgr = cv2.cvtColor((out_rgb * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), out_bgr)

        # Save mask if requested
        if mask_dir is not None:
            m = mask if mask.ndim == 2 else mask[:, :, 0]
            mask_path = mask_dir / img_path.name
            cv2.imwrite(str(mask_path), (m.clip(0, 1) * 255.0).astype(np.uint8))

        # Compute metrics
        metrics_data = {"image": img_path.name}
        
        if gt_dir is not None and gt_dir.exists():
            gt_path = gt_dir / img_path.name
            if gt_path.exists():
                gt_bgr = cv2.imread(str(gt_path))
                gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Convert to tensors for PSNR
                out_tensor = torch.from_numpy(out_rgb).permute(2, 0, 1)
                gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1)
                
                # Compute metrics
                psnr = MetricsComputer.compute_psnr(out_tensor, gt_tensor, max_value=1.0)
                ssim = MetricsComputer.compute_ssim(out_rgb, gt_rgb, data_range=1.0, channel_axis=2)
                
                metrics_data["psnr"] = float(psnr)
                metrics_data["ssim"] = float(ssim)
                
                if verbose:
                    print(f"  {img_path.name}: PSNR={psnr:.4f} | SSIM={ssim:.4f}")
        
        metrics_list.append(metrics_data)
        processed_images.append(img_path.name)

    # Save summary
    summary = {
        "total_processed": len(processed_images),
        "images": processed_images,
        "metrics": metrics_list,
        "output_dir": str(output_dir.resolve()),
        "mask_dir": str(mask_dir.resolve()) if mask_dir else None,
    }

    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nProcessing Complete!")
        print(f"  Images processed: {summary['total_processed']}")
        print(f"  Outputs saved to: {output_dir.resolve()}")
        if mask_dir:
            print(f"  Masks saved to: {mask_dir.resolve()}")
        print(f"  Summary saved to: {summary_file.resolve()}")


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

    p_eval = sub.add_parser(
        "eval",
        help="Run inference on random images and save outputs.",
    )
    p_eval.add_argument(
        "--input-dir",
        required=True,
        help="Directory with shadow images",
    )
    p_eval.add_argument(
        "--output-dir",
        default="outputs/removed",
        help="Directory to save shadow-removed outputs",
    )
    p_eval.add_argument(
        "--mask-dir",
        default=None,
        metavar="PATH",
        help="Optional directory to save predicted shadow masks",
    )
    p_eval.add_argument(
        "--gt-dir",
        default=None,
        help="Optional directory with ground truth images for metrics",
    )
    p_eval.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of random images to process (default: 20)",
    )
    p_eval.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    p_eval.add_argument(
        "--detector-weights",
        default=None,
        help="Detector weights .pth (default: models/shadow_detection.pth)",
    )
    p_eval.add_argument(
        "--removal-weights",
        default=None,
        help="Removal weights .pth (default: models/shadow_removal.pth)",
    )
    p_eval.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for inference (auto by default)",
    )
    p_eval.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress",
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
    elif args.command == "eval":
        run_inference_batch(
            args.input_dir,
            args.output_dir,
            mask_dir=args.mask_dir,
            detector_weights=args.detector_weights,
            removal_weights=args.removal_weights,
            device=args.device,
            num_images=args.num_images,
            seed=args.seed,
            gt_dir=args.gt_dir,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
