from typing import Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def to_classes(y_pred: np.ndarray) -> np.ndarray:
    # Use a more dynamic approach to classification
    # Instead of rounding, use thresholds to create more diverse predictions
    y_classes = np.ones_like(y_pred, dtype=int)
    
    # Define thresholds for different risk levels
    low_high = 1.67  # Threshold between low (1) and medium (2) risk
    high_severe = 2.33  # Threshold between medium (2) and high (3) risk
    
    # Apply thresholds
    y_classes[y_pred > low_high] = 2
    y_classes[y_pred > high_severe] = 3
    
    return y_classes


def classification_report_123(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    y_hat = to_classes(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="macro", zero_division=0)
    return float(precision), float(recall), float(f1)