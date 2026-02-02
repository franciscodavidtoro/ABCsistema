import numpy as np
import cv2


class HSVExtractor:
    """
    Extractor HSV (Histogramas de Color).
    
    Extrae características basadas en la distribución de color (HSV)
    en diferentes regiones de la imagen.
    
    Attributes:
        h_bins (int): Número de bins para el histograma de Hue.
        s_bins (int): Número de bins para el histograma de Saturation.
        v_bins (int): Número de bins para el histograma de Value.
        n_regions (tuple): División espacial de la imagen (filas, columnas).
        output_dim (int): Dimensionalidad del vector de salida.
    """
    
    def __init__(self, h_bins: int = 32, s_bins: int = 16, v_bins: int = 16,
                 n_regions: tuple = (4, 4), output_dim: int = 2048):
        """
        Inicializa el extractor HSV.
        
        Args:
            h_bins (int): Bins para Hue (0-180). Default: 32.
            s_bins (int): Bins para Saturation (0-255). Default: 16.
            v_bins (int): Bins para Value (0-255). Default: 16.
            n_regions (tuple): División espacial (filas, columnas). Default: (4, 4).
            output_dim (int): Dimensionalidad del vector de salida. Default: 2048.
        """
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.n_regions = n_regions
        self.output_dim = output_dim
        
        # Validar parámetros
        if h_bins < 1 or s_bins < 1 or v_bins < 1:
            raise ValueError("Los bins deben ser valores positivos")
        if n_regions[0] < 1 or n_regions[1] < 1:
            raise ValueError("n_regions debe tener valores positivos")
        
        # Calcular dimensionalidad esperada por región
        self.hist_size_per_region = h_bins * s_bins * v_bins
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características HSV de una imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada (H, W, C) en formato BGR o RGB.
        
        Returns:
            np.ndarray: Vector de características HSV de dimensión (output_dim,).
        """
        # Validar formato de imagen
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser un numpy.ndarray")
        
        if image.size == 0:
            raise ValueError("La imagen está vacía")
        
        # Asegurar que sea imagen en color
        if len(image.shape) == 2:
            # Convertir escala de grises a BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("La imagen debe ser BGR/RGB de 3 canales")
        
        # Convertir a espacio de color HSV
        hsv_image = self._convert_to_hsv(image)
        
        # Calcular histogramas espaciales
        feature_vector = self._compute_spatial_histograms(hsv_image)
        
        # Normalizar vector resultante
        feature_vector = self._normalize_histogram(feature_vector)
        
        # Redimensionar a output_dim
        if len(feature_vector) > self.output_dim:
            feature_vector = feature_vector[:self.output_dim]
        elif len(feature_vector) < self.output_dim:
            padding = np.zeros(self.output_dim - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        
        return feature_vector.astype(np.float32)
    
    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extrae características HSV de un lote de imágenes.
        
        Args:
            images (list): Lista de imágenes (numpy arrays).
        
        Returns:
            np.ndarray: Matriz de características de dimensión (N, output_dim).
        """
        if not isinstance(images, (list, np.ndarray)):
            raise TypeError("images debe ser una lista o numpy.ndarray")
        
        if len(images) == 0:
            raise ValueError("La lista de imágenes está vacía")
        
        print(f"[DEBUG HSVExtractor] Iniciando extract_batch con {len(images)} imágenes")
        import time
        total_start = time.time()
        
        # Extraer características de cada imagen
        features = []
        for idx, img in enumerate(images):
            if idx % 10 == 0:
                elapsed = time.time() - total_start
                print(f"[DEBUG HSVExtractor] Procesando imagen {idx+1}/{len(images)} (tiempo: {elapsed:.1f}s)")
            feat = self.extract(img)
            features.append(feat)
        
        total_elapsed = time.time() - total_start
        print(f"[DEBUG HSVExtractor] Batch completado en {total_elapsed:.2f}s ({total_elapsed/len(images):.2f}s/imagen)")
        
        # Apilar resultados en una matriz
        return np.array(features, dtype=np.float32)
    
    def _convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Convierte imagen BGR/RGB a espacio de color HSV.
        
        Args:
            image (np.ndarray): Imagen en formato BGR o RGB.
        
        Returns:
            np.ndarray: Imagen en espacio de color HSV.
        """
        # OpenCV asume BGR por defecto
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv
    
    def _compute_spatial_histograms(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Calcula histogramas espaciales en el espacio HSV.
        
        Args:
            hsv_image (np.ndarray): Imagen en espacio HSV.
        
        Returns:
            np.ndarray: Vector concatenado de histogramas por región.
        """
        h, w = hsv_image.shape[:2]
        rows, cols = self.n_regions
        
        # Calcular tamaño de cada región
        region_h = h // rows
        region_w = w // cols
        
        histograms = []
        
        # Para cada región espacial
        for i in range(rows):
            for j in range(cols):
                # Calcular límites de la región
                y_start = i * region_h
                y_end = (i + 1) * region_h if i < rows - 1 else h
                x_start = j * region_w
                x_end = (j + 1) * region_w if j < cols - 1 else w
                
                # Extraer región
                region = hsv_image[y_start:y_end, x_start:x_end]
                
                # Calcular histograma 3D (H, S, V) usando cv2.calcHist
                hist = cv2.calcHist(
                    [region],
                    [0, 1, 2],  # Canales H, S, V
                    None,
                    [self.h_bins, self.s_bins, self.v_bins],
                    [0, 180, 0, 256, 0, 256]  # Rangos: H[0,180], S[0,256], V[0,256]
                )
                
                # Aplanar histograma y añadir a la lista
                hist_flat = hist.flatten()
                histograms.append(hist_flat)
        
        # Concatenar todos los histogramas
        concatenated = np.concatenate(histograms)
        
        return concatenated
    
    def _normalize_histogram(self, histogram: np.ndarray) -> np.ndarray:
        """
        Normaliza un histograma.
        
        Args:
            histogram (np.ndarray): Histograma a normalizar.
        
        Returns:
            np.ndarray: Histograma normalizado.
        """
        # Normalizar usando L2 norm
        norm = np.linalg.norm(histogram)
        
        if norm > 0:
            normalized = histogram / (norm + 1e-6)
        else:
            normalized = histogram
        
        return normalized