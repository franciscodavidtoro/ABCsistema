from __future__ import annotations

from typing import Tuple, Optional, Union, List
import numpy as np
import cv2

# Intentar usar scikit-image (más robusto y optimizado para HOG)
try:
    from skimage.feature import hog as sk_hog
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


class HOGExtractor:
    """Extracción de HOG con opciones para resize y diferentes implementaciones.

    Args:
        orientations: número de bins para el histograma (por defecto 9)
        pixels_per_cell: tamaño (alto, ancho) en píxeles de cada celda (por defecto (8,8))
        cells_per_block: número de celdas por bloque (alto, ancho) (por defecto (2,2))
        image_size: si se especifica (H, W) las imágenes se redimensionan a este
                    tamaño. Por defecto (128, 64) para evitar vectores gigantes.
        output_dim: dimensión fija de salida. Si None, se calculará automáticamente
                    cuando se llame a `set_image_size` o tras la primera extracción
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        image_size: Optional[Tuple[int, int]] = (128, 64),
        output_dim: Optional[int] = None,
    ) -> None:
        # Parámetros
        if orientations < 1:
            raise ValueError("orientations debe ser >= 1")
        if pixels_per_cell[0] < 1 or pixels_per_cell[1] < 1:
            raise ValueError("pixels_per_cell debe contener valores positivos")
        if cells_per_block[0] < 1 or cells_per_block[1] < 1:
            raise ValueError("cells_per_block debe contener valores positivos")

        self.orientations = int(orientations)
        self.pixels_per_cell = (int(pixels_per_cell[0]), int(pixels_per_cell[1]))
        self.cells_per_block = (int(cells_per_block[0]), int(cells_per_block[1]))
        self.image_size = image_size  # (H, W) o None
        self.output_dim = output_dim

        # Si se especifica image_size, calcular output_dim previsible
        if self.image_size is not None:
            self.output_dim = self.compute_output_dim(self.image_size)

    def set_image_size(self, image_size: Tuple[int, int]) -> None:
        """Establece tamaño fijo de imagen (H, W) y recalcula `output_dim`."""
        if image_size[0] < 1 or image_size[1] < 1:
            raise ValueError("image_size debe contener valores positivos")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.output_dim = self.compute_output_dim(self.image_size)

    def compute_output_dim(self, image_size: Tuple[int, int]) -> int:
        """Calcula la dimensión del vector HOG para una `image_size` dada."""
        h, w = image_size
        cell_h, cell_w = self.pixels_per_cell
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w
        block_h, block_w = self.cells_per_block
        n_blocks_y = max(0, n_cells_y - block_h + 1)
        n_blocks_x = max(0, n_cells_x - block_w + 1)
        features_per_block = self.orientations * block_h * block_w
        return int(n_blocks_y * n_blocks_x * features_per_block)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extrae el vector HOG de una imagen.

        - Si `image_size` está definida, la imagen se redimensiona a ese tamaño.
        - Si `skimage` está presente se usa su implementación (más estable).

        Retorna un vector 1D de tipo float32 con longitud `output_dim`.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser un numpy.ndarray")
        if image.size == 0:
            raise ValueError("La imagen está vacía")

        # Convertir a escala de grises si es necesario
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Redimensionar si fue solicitado para garantizar dimensión fija
        if self.image_size is not None:
            gray = self._resize_image(gray, self.image_size)

        # Usar scikit-image si está disponible
        if SKIMAGE_AVAILABLE:
            # sk_hog retorna array 1D con features
            features = sk_hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True,
            )
            features = np.asarray(features, dtype=np.float32)
        else:
            # Fallback: implementación manual
            magnitude, orientation = self._compute_gradients(gray)
            histograms = self._build_histograms(magnitude, orientation)
            normalized = self._normalize_blocks(histograms)
            features = self._flatten_features(normalized).astype(np.float32)

        # Si no tenemos output_dim definido, lo calculamos a partir de las características
        if self.output_dim is None:
            self.output_dim = features.shape[0]

        # Ajustar longitud a output_dim: truncar o rellenar con ceros
        if features.shape[0] > self.output_dim:
            features = features[: self.output_dim]
        elif features.shape[0] < self.output_dim:
            pad = np.zeros(self.output_dim - features.shape[0], dtype=np.float32)
            features = np.concatenate([features, pad])

        return features.astype(np.float32)

    def extract_batch(self, images: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Extrae HOG para un lote de imágenes y retorna matriz (N, output_dim)."""
        if not isinstance(images, (list, np.ndarray)):
            raise TypeError("images debe ser lista o numpy.ndarray")
        if len(images) == 0:
            raise ValueError("La lista de imágenes está vacía")

        features = [self.extract(img) for img in images]
        return np.stack(features, axis=0).astype(np.float32)

    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Redimensiona imagen manteniendo tipo y usando interpolación adecuada."""
        h, w = size
        if image.shape[0] == h and image.shape[1] == w:
            return image
        # Si se reduce tamaño, usar INTER_AREA; si se aumenta, INTER_LINEAR
        interpolation = cv2.INTER_AREA if (image.shape[0] > h or image.shape[1] > w) else cv2.INTER_LINEAR
        resized = cv2.resize(image, (w, h), interpolation=interpolation)
        return resized

    def _compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula magnitud y orientación de gradientes (en grados 0..180).

        Se usa Sobel con precisión float32.
        """
        image = image.astype(np.float32)
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        magnitude = np.hypot(gx, gy)
        orientation = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0
        return magnitude, orientation

    def _build_histograms(self, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Crea histogramas por celda ponderados por magnitud.

        Devuelve array shape (n_cells_y, n_cells_x, orientations).
        """
        h, w = magnitude.shape
        cell_h, cell_w = self.pixels_per_cell
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w

        # Inicializar
        hist = np.zeros((n_cells_y, n_cells_x, self.orientations), dtype=np.float32)

        bin_size = 180.0 / float(self.orientations)

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                y0, y1 = i * cell_h, (i + 1) * cell_h
                x0, x1 = j * cell_w, (j + 1) * cell_w
                cell_mag = magnitude[y0:y1, x0:x1]
                cell_ori = orientation[y0:y1, x0:x1]

                # Flatten and assign
                flat_mag = cell_mag.ravel()
                flat_ori = cell_ori.ravel()
                # Compute bin indices clipped to [0, orientations-1]
                bins = (flat_ori / bin_size).astype(int) % self.orientations
                for b_idx, m in zip(bins, flat_mag):
                    hist[i, j, b_idx] += m
        return hist

    def _normalize_blocks(self, histograms: np.ndarray) -> np.ndarray:
        """Normaliza bloques adyacentes y devuelve lista de vectores por bloque."""
        n_cells_y, n_cells_x, _ = histograms.shape
        block_h, block_w = self.cells_per_block
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1

        if n_blocks_y <= 0 or n_blocks_x <= 0:
            # No hay bloques completos, retornar histograms flatten
            return histograms.reshape(-1,)

        blocks = []
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = histograms[i : i + block_h, j : j + block_w, :].flatten()
                norm = np.linalg.norm(block)
                if norm > 0:
                    block = block / (norm + 1e-6)
                blocks.append(block)
        return np.asarray(blocks, dtype=np.float32)

    def _flatten_features(self, normalized_histograms: np.ndarray) -> np.ndarray:
        """Apila vectores normalizados en un vector 1D final."""
        if normalized_histograms.ndim == 1:
            return normalized_histograms.ravel()
        return normalized_histograms.ravel()
