"""
Módulo para entrenamiento del clasificador SVM.

Implementa el pipeline de entrenamiento incluyendo preparación de datos,
validación cruzada y ajuste de hiperparámetros.
"""

from __future__ import annotations

import time
from typing import Tuple, Optional, Any

import numpy as np

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from .svm_model import SVMModel


class ModelTrainer:
    """Entrenador que encapsula preparación de datos y entrenamiento."""

    def __init__(self, svm_model: Optional[SVMModel] = None, train_test_split: float = 0.8):
        if svm_model is None:
            self.svm_model = SVMModel()
        else:
            self.svm_model = svm_model

        if not (0.0 < train_test_split < 1.0):
            raise ValueError("train_test_split debe estar entre 0 y 1")
        self.train_test_split = train_test_split
        self.scaler = None

    def prepare_data(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

        # Fill NaNs
        if np.isnan(features).any():
            features = np.nan_to_num(features)

        scaler = None
        if normalize:
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
            else:
                mean = features.mean(axis=0)
                std = features.std(axis=0)
                std[std == 0] = 1.0
                features = (features - mean) / std
                scaler = {'mean': mean, 'std': std}

        self.scaler = scaler
        return features, labels, scaler

    def split_data(self, features: np.ndarray, labels: np.ndarray):
        if SKLEARN_AVAILABLE:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=1.0 - self.train_test_split, stratify=labels, random_state=42
            )
        else:
            # Simple split keeping class proportions
            n = features.shape[0]
            idx = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(idx)
            cut = int(n * self.train_test_split)
            train_idx = idx[:cut]
            test_idx = idx[cut:]
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
        return X_train, X_test, y_train, y_test

    def train(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True) -> Tuple[SVMModel, dict]:
        start = time.time()
        X, y, scaler = self.prepare_data(features, labels, normalize=normalize)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        stats_train = self.svm_model.train(X_train, y_train)

        # Evaluate on test
        try:
            preds = self.svm_model.predict(X_test)
            if SKLEARN_AVAILABLE:
                acc = accuracy_score(y_test, preds)
            else:
                acc = (preds == y_test).mean()
        except Exception:
            acc = None

        stats = {
            **stats_train,
            'test_accuracy': float(acc) if acc is not None else None,
            'train_elapsed_sec': time.time() - start
        }
        return self.svm_model, stats

    def cross_validate(self, features: np.ndarray, labels: np.ndarray, n_folds: int = 5):
        if not SKLEARN_AVAILABLE:
            raise NotImplementedError("cross_validate requiere scikit-learn")
        X, y, _ = self.prepare_data(features, labels)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        # Create a fresh estimator for cross val
        # Use SVC with the trainer's svm_model params
        from sklearn.svm import SVC
        estimator = SVC(kernel=self.svm_model.kernel, C=self.svm_model.C)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy')
        return {'fold_accuracies': list(scores), 'mean_accuracy': float(scores.mean())}

    def tune_hyperparameters(self, features: np.ndarray, labels: np.ndarray, param_grid: dict, cv: int = 3):
        if not SKLEARN_AVAILABLE:
            raise NotImplementedError("tune_hyperparameters requiere scikit-learn")
        X, y, _ = self.prepare_data(features, labels)
        base = SVMModel()
        from sklearn.svm import SVC
        estimator = SVC()
        gs = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        gs.fit(X, y)
        return {'best_params': gs.best_params_, 'best_score': float(gs.best_score_)}
