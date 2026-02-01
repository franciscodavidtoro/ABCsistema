"""
Módulo para evaluación del modelo de reidentificación.

Implementa métricas y análisis de desempeño del clasificador SVM
y del sistema de reidentificación en general.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class ModelEvaluator:
    """Clase para evaluar el desempeño del modelo de reidentificación."""

    def __init__(self, svm_model: Any = None):
        self.svm_model = svm_model

    def evaluate(self, model: Any, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Evalúa y retorna métricas básicas: accuracy, precision, recall, f1, confusion matrix."""
        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

        preds, confidences = model.predict_with_confidence(features)

        # Calculate accuracy
        accuracy = self.calculate_accuracy(preds, labels)

        # Precision/recall/F1
        precision = None
        recall = None
        f1 = None
        conf_mat = None

        if SKLEARN_AVAILABLE:
            precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(labels, preds, zero_division=0)
            precision = precision_arr.tolist()
            recall = recall_arr.tolist()
            f1 = f1_arr.tolist()
            conf_mat = confusion_matrix(labels, preds).tolist()
            mean_precision = float(np.mean(precision_arr))
            mean_recall = float(np.mean(recall_arr))
            mean_f1 = float(np.mean(f1_arr))
        else:
            # Simple calculations per class
            classes = list(np.unique(np.concatenate((labels, preds))))
            precision_arr = []
            recall_arr = []
            f1_arr = []
            for c in classes:
                tp = int(((preds == c) & (labels == c)).sum())
                fp = int(((preds == c) & (labels != c)).sum())
                fn = int(((preds != c) & (labels == c)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1s = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                precision_arr.append(prec)
                recall_arr.append(rec)
                f1_arr.append(f1s)
            precision = precision_arr
            recall = recall_arr
            f1 = f1_arr
            mean_precision = float(np.mean(precision_arr))
            mean_recall = float(np.mean(recall_arr))
            mean_f1 = float(np.mean(f1_arr))
            # build confusion matrix
            class_to_idx = {c: i for i, c in enumerate(classes)}
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(labels, preds):
                cm[class_to_idx[t], class_to_idx[p]] += 1
            conf_mat = cm.tolist()

        metrics = {
            'accuracy': float(accuracy),
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'confusion_matrix': conf_mat,
            'num_samples': int(len(labels))
        }
        return metrics

    def calculate_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        if SKLEARN_AVAILABLE:
            return float(accuracy_score(labels, predictions))
        else:
            return float((predictions == labels).mean())

    # The other helper methods (precision/recall/f1/confusion matrix) are implemented within evaluate
    # but kept for API completeness

    def calculate_precision_recall(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        _, precision_arr, recall_arr, _ = precision_recall_fscore_support(labels, predictions, zero_division=0) if SKLEARN_AVAILABLE else (None, [], [], None)
        return {'precision_per_class': precision_arr.tolist() if hasattr(precision_arr, 'tolist') else precision_arr,
                'recall_per_class': recall_arr.tolist() if hasattr(recall_arr, 'tolist') else recall_arr}

    def calculate_f1_score(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        _, _, f1_arr, _ = precision_recall_fscore_support(labels, predictions, zero_division=0) if SKLEARN_AVAILABLE else (None, None, [], None)
        return {'f1_per_class': f1_arr.tolist() if hasattr(f1_arr, 'tolist') else f1_arr}

    def generate_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray):
        if SKLEARN_AVAILABLE:
            return confusion_matrix(labels, predictions)
        else:
            classes = list(np.unique(np.concatenate((labels, predictions))))
            class_to_idx = {c: i for i, c in enumerate(classes)}
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(labels, predictions):
                cm[class_to_idx[t], class_to_idx[p]] += 1
            return cm

    def generate_report(self, predictions: np.ndarray, labels: np.ndarray) -> str:
        metrics = self.evaluate(self.svm_model, predictions, labels) if self.svm_model is not None else {}
        return str(metrics)

    def plot_metrics(self, predictions: np.ndarray, labels: np.ndarray, output_path: str = None):
        # Lightweight: no plotting unless matplotlib is available; placeholder
        try:
            import matplotlib.pyplot as plt
            cm = self.generate_confusion_matrix(predictions, labels)
            fig, ax = plt.subplots()
            cax = ax.matshow(cm)
            fig.colorbar(cax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            if output_path:
                fig.savefig(output_path)
                return {'saved': output_path}
            return {'fig': fig}
        except Exception:
            return {'status': 'plotting not available'}
