"""Metrics computation and tracking for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


@dataclass
class MetricsTracker:
    """Track and compute metrics during training/validation."""

    epoch: int = 0
    batch: int = 0
    loss: float = 0.0
    losses: list[float] = field(default_factory=list)
    
    def update_loss(self, loss_value: float) -> None:
        """Update loss metric."""
        self.loss = loss_value
        self.losses.append(loss_value)
    
    def reset(self) -> None:
        """Reset metrics for new epoch."""
        self.loss = 0.0
        self.losses = []
    
    def avg_loss(self) -> float:
        """Compute average loss."""
        return np.mean(self.losses) if self.losses else 0.0
    
    def summary(self) -> str:
        """Return formatted summary of current metrics."""
        avg = self.avg_loss()
        return f"Epoch [{self.epoch}] Loss: {avg:.4f}"


class MetricsComputer:
    """Compute various evaluation metrics."""

    @staticmethod
    def compute_loss(
        criterion,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute loss between predictions and targets."""
        loss = criterion(predictions, targets)
        return loss.item()

    @staticmethod
    def compute_binary_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """
        Compute binary classification metrics (Accuracy, Precision, Recall, F1).
        
        Args:
            predictions: Model predictions (logits or probabilities), shape (N, H, W) or (N, 1, H, W)
            targets: Ground truth labels, shape (N, H, W) or (N, 1, H, W)
            threshold: Binary threshold for predictions
        
        Returns:
            Dict with keys: 'accuracy', 'precision', 'recall', 'f1'
        """
        # Ensure same shape
        if predictions.shape != targets.shape:
            predictions = predictions.squeeze()
            targets = targets.squeeze()
        
        # Apply threshold
        pred_binary = (predictions > threshold).float()
        target_binary = targets.float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        tgt_flat = target_binary.view(-1)
        
        # Compute TP, FP, TN, FN
        tp = (pred_flat * tgt_flat).sum().item()
        fp = (pred_flat * (1 - tgt_flat)).sum().item()
        tn = ((1 - pred_flat) * (1 - tgt_flat)).sum().item()
        fn = ((1 - pred_flat) * tgt_flat).sum().item()
        
        # Metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def compute_iou(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Intersection over Union (IoU).
        
        Args:
            predictions: Model predictions, shape (N, H, W) or (N, 1, H, W)
            targets: Ground truth labels, shape (N, H, W) or (N, 1, H, W)
            threshold: Binary threshold for predictions
        
        Returns:
            IoU score (0-1)
        """
        # Ensure same shape
        if predictions.shape != targets.shape:
            predictions = predictions.squeeze()
            targets = targets.squeeze()
        
        # Apply threshold
        pred_binary = (predictions > threshold).float()
        target_binary = targets.float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        tgt_flat = target_binary.view(-1)
        
        # IoU calculation
        intersection = (pred_flat * tgt_flat).sum().item()
        union = (pred_flat + tgt_flat - pred_flat * tgt_flat).sum().item()
        
        iou = intersection / (union + 1e-8)
        return iou

    @staticmethod
    def compute_dice(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Dice coefficient.
        
        Args:
            predictions: Model predictions, shape (N, H, W) or (N, 1, H, W)
            targets: Ground truth labels, shape (N, H, W) or (N, 1, H, W)
            threshold: Binary threshold for predictions
        
        Returns:
            Dice coefficient (0-1)
        """
        # Ensure same shape
        if predictions.shape != targets.shape:
            predictions = predictions.squeeze()
            targets = targets.squeeze()
        
        # Apply threshold
        pred_binary = (predictions > threshold).float()
        target_binary = targets.float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        tgt_flat = target_binary.view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * tgt_flat).sum().item()
        dice = (2 * intersection) / (pred_flat.sum().item() + tgt_flat.sum().item() + 1e-8)
        
        return dice

    @staticmethod
    def compute_mae(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute Mean Absolute Error.
        
        Args:
            predictions: Model predictions (0-1 or 0-255)
            targets: Ground truth labels (same scale)
        
        Returns:
            MAE value
        """
        mae = F.l1_loss(predictions, targets)
        return mae.item()

    @staticmethod
    def compute_mse(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute Mean Squared Error.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
        
        Returns:
            MSE value
        """
        mse = F.mse_loss(predictions, targets)
        return mse.item()

    @staticmethod
    def compute_rmse(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute Root Mean Squared Error.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
        
        Returns:
            RMSE value
        """
        mse = F.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse)
        return rmse.item()

    @staticmethod
    def compute_psnr(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        max_value: float = 1.0,
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            predictions: Model predictions (0-1 or 0-255)
            targets: Ground truth labels (same scale)
            max_value: Maximum pixel value (1.0 for 0-1 range, 255.0 for 0-255 range)
        
        Returns:
            PSNR value in dB
        """
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
        """
        Compute Structural Similarity Index (SSIM).
        
        Args:
            predictions: Model predictions (0-1 or 0-255 numpy array)
            targets: Ground truth labels (same scale, same shape)
            data_range: Maximum value range (1.0 for 0-1, 255.0 for 0-255)
            channel_axis: Axis for channels (None for grayscale, 2 for RGB images)
        
        Returns:
            SSIM value (-1 to 1, higher is better)
        """
        return structural_similarity(
            targets,
            predictions,
            data_range=data_range,
            channel_axis=channel_axis,
        )

    @staticmethod
    def compute_all_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        criterion=None,
        threshold: float = 0.5,
        max_value: float = 1.0,
    ) -> dict:
        """
        Compute all available metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            criterion: Loss criterion (optional)
            threshold: Binary threshold for classification metrics
            max_value: Maximum pixel value for PSNR
        
        Returns:
            Dict with all computed metrics
        """
        metrics = {}
        
        if criterion is not None:
            metrics["loss"] = MetricsComputer.compute_loss(criterion, predictions, targets)
        
        metrics["mae"] = MetricsComputer.compute_mae(predictions, targets)
        metrics["mse"] = MetricsComputer.compute_mse(predictions, targets)
        metrics["rmse"] = MetricsComputer.compute_rmse(predictions, targets)
        metrics["psnr"] = MetricsComputer.compute_psnr(predictions, targets, max_value)
        
        # Binary metrics (for segmentation)
        binary_metrics = MetricsComputer.compute_binary_metrics(predictions, targets, threshold)
        metrics.update(binary_metrics)
        
        metrics["iou"] = MetricsComputer.compute_iou(predictions, targets, threshold)
        metrics["dice"] = MetricsComputer.compute_dice(predictions, targets, threshold)
        
        return metrics


def format_metrics(metrics: dict, decimals: int = 4) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.{decimals}f}")
    return " | ".join(lines)
