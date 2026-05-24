from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


class MetricsComputer:
    @staticmethod
    def compute_psnr(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        max_value: float = 1.0,
    ) -> float:
        mse = F.mse_loss(predictions, targets)
        if mse == 0:
            return 100.0  # Perfect match

        psnr = 20 * torch.log10(torch.tensor(max_value) / torch.sqrt(mse))
        return psnr.item()

    @staticmethod
    def compute_ssim(
        predictions: np.ndarray,
        targets: np.ndarray,
        data_range: float = 1.0,
        channel_axis: int | None = None,
    ) -> float:
        return structural_similarity(
            targets,
            predictions,
            data_range=data_range,
            channel_axis=channel_axis,
        )


def format_metrics(metrics: dict, decimals: int = 4) -> str:
    lines = []
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.{decimals}f}")
    return " | ".join(lines)
