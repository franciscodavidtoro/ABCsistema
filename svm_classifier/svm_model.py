"""
Módulo del modelo SVM para clasificación de características corporales.

Implementa un clasificador Support Vector Machine entrenado para
diferenciar personas basándose en sus características físicas.

Se utiliza scikit-learn si está disponible; si no, se proporciona un
fallback sencillo basado en centroides de clase.
"""

from __future__ import annotations

import os
import time
import pickle
from typing import Tuple, Any

import numpy as np

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class SVMModel:
    """
    Clase que encapsula el modelo SVM para reidentificación.
    """

    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        self.kernel = kernel
        self.C = C
        self.clf = None  # sklearn estimator or None
        self.label_encoder = None
        self.is_trained = False
        self.class_centroids = None
        self.classes_ = None

        if SKLEARN_AVAILABLE:
            # Only instantiate classifier when training to avoid import-time overhead
            self._sklearn = True
        else:
            self._sklearn = False

    def train(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """Entrena el modelo con features y labels.

        Returns estadísticas básicas del entrenamiento.
        """
        start = time.time()

        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features y labels deben tener el mismo número de muestras")

        # Encode labels
        if SKLEARN_AVAILABLE:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
            self.classes_ = list(self.label_encoder.classes_)

            self.clf = SVC(kernel=self.kernel, C=self.C, probability=True)
            self.clf.fit(features, y)
            self.is_trained = True
        else:
            # Fallback: nearest centroid
            unique = np.unique(labels)
            centroids = {}
            for u in unique:
                centroids[u] = features[labels == u].mean(axis=0)
            self.class_centroids = centroids
            self.classes_ = list(unique)
            self.label_encoder = None
            self.is_trained = True

        elapsed = time.time() - start

        stats = {
            'num_samples': int(features.shape[0]),
            'num_features': int(features.shape[1]),
            'num_classes': int(len(self.classes_)),
            'train_time_sec': float(elapsed)
        }
        return stats

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Retorna predicciones en forma de etiquetas originales."""
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado")

        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if SKLEARN_AVAILABLE and self.clf is not None:
            y_pred = self.clf.predict(features)
            # decode
            return self.label_encoder.inverse_transform(y_pred)
        else:
            # nearest centroid
            preds = []
            centroids = self.class_centroids
            classes = self.classes_
            for x in features:
                best = None
                best_c = None
                for c in classes:
                    d = np.linalg.norm(x - centroids[c])
                    if best is None or d < best:
                        best = d
                        best_c = c
                preds.append(best_c)
            return np.array(preds)

    def predict_with_confidence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predice etiquetas y retorna score de confianza (0..1)"""
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado")

        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if SKLEARN_AVAILABLE and self.clf is not None:
            probs = self.clf.predict_proba(features)
            pred_idx = probs.argmax(axis=1)
            preds = self.label_encoder.inverse_transform(pred_idx)
            confidences = probs.max(axis=1)
            return preds, confidences
        else:
            preds = []
            confidences = []
            centroids = self.class_centroids
            classes = self.classes_
            for x in features:
                best = None
                best_c = None
                distances = []
                for c in classes:
                    d = np.linalg.norm(x - centroids[c])
                    distances.append((c, d))
                    if best is None or d < best:
                        best = d
                        best_c = c
                # transform distances into a pseudo-confidence
                ds = np.array([d for (_, d) in distances])
                # avoid division by zero
                inv = 1.0 / (ds + 1e-8)
                conf_scores = inv / inv.sum()
                # confidence of predicted class
                conf = conf_scores[[i for i,(cc,_) in enumerate(distances) if cc==best_c][0]]
                preds.append(best_c)
                confidences.append(float(conf))
            return np.array(preds), np.array(confidences)

    def save(self, model_path: str) -> bool:
        """Serializa y guarda el objeto SVMModel completo."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True) if os.path.dirname(model_path) else None
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        return True

    @classmethod
    def load(cls, model_path: str) -> 'SVMModel':
        """Carga y retorna una instancia de SVMModel desde archivo."""
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, SVMModel):
            raise TypeError("El archivo no contiene un SVMModel")
        return obj

    # Backwards compatible aliases
    def save_model(self, model_path: str) -> bool:
        return self.save(model_path)

    @classmethod
    def load_model(cls, model_path: str) -> 'SVMModel':
        return cls.load(model_path)

    def get_person_id(self, class_index: Any) -> Any:
        """Retorna la etiqueta original de una clase dado su índice interno."""
        if SKLEARN_AVAILABLE and self.label_encoder is not None:
            # class_index expected to be integer index
            return self.label_encoder.classes_[int(class_index)]
        else:
            # assume class_index is index into classes_
            return self.classes_[int(class_index)]

