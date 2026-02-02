import numpy as np
import cv2
from skimage.feature import local_binary_pattern


class LBPExtractor:
    """
    Extractor LBP (Local Binary Patterns) optimizado.
    
    Utiliza la implementación eficiente de scikit-image para extraer
    características de textura basadas en patrones binarios locales.
    
    Attributes:
        radius (int): Radio del operador LBP.
        n_points (int): Número de puntos de muestreo alrededor del pixel central.
        method (str): Método de cálculo ('default', 'ror', 'uniform', 'nri_uniform', 'var').
        grid_size (tuple): Tamaño de la grilla para histogramas espaciales.
        output_dim (int): Dimensionalidad del vector de salida.
    """
    
    def __init__(self, radius: int = 1, n_points: int = 8, method: str = 'uniform',
                 grid_size: tuple = (4, 4), output_dim: int = 256):
        """
        Inicializa el extractor LBP.
        
        Args:
            radius (int): Radio del operador LBP. Default: 1.
            n_points (int): Número de puntos de muestreo. Default: 8.
            method (str): Método de cálculo ('uniform', 'default', 'ror', etc). Default: 'uniform'.
            grid_size (tuple): Tamaño de la grilla (filas, columnas). Default: (4, 4).
            output_dim (int): Dimensionalidad del vector de salida. Default: 256.
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.grid_size = grid_size
        self.output_dim = output_dim
        
        # Validar parámetros
        if radius < 1:
            raise ValueError("radius debe ser al menos 1")
        if n_points < 1:
            raise ValueError("n_points debe ser al menos 1")
        if grid_size[0] < 1 or grid_size[1] < 1:
            raise ValueError("grid_size debe tener valores positivos")
        
        # Calcular número de bins según método
        if method == 'uniform':
            # Para patrones uniformes: n_points + 2 bins
            self.n_bins = n_points + 2
        elif method == 'nri_uniform':
            # Non-rotation invariant uniform
            self.n_bins = n_points * (n_points - 1) + 3
        else:
            # Para método default: 2^n_points bins
            self.n_bins = 2 ** n_points
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características LBP de una imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada (H, W, C) o (H, W).
        
        Returns:
            np.ndarray: Vector de características LBP de dimensión (output_dim,).
        """
        # Validar imagen
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser un numpy.ndarray")
        
        if image.size == 0:
            raise ValueError("La imagen está vacía")
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Redimensionar para acelerar el procesamiento si la imagen es muy grande
        max_size = 128
        h, w = gray.shape
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calcular imagen LBP usando skimage (MUY rápido, implementado en C)
        lbp_image = local_binary_pattern(gray, self.n_points, self.radius, method=self.method)
        
        # Calcular histogramas espaciales
        feature_vector = self._get_spatial_histograms(lbp_image)
        
        # Normalizar L2
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / (norm + 1e-6)
        
        # Ajustar a output_dim
        if len(feature_vector) > self.output_dim:
            feature_vector = feature_vector[:self.output_dim]
        elif len(feature_vector) < self.output_dim:
            padding = np.zeros(self.output_dim - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        
        return feature_vector.astype(np.float32)
    
    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extrae características LBP de un lote de imágenes.
        
        Args:
            images (list): Lista de imágenes (numpy arrays).
        
        Returns:
            np.ndarray: Matriz de características de dimensión (N, output_dim).
        """
        if not isinstance(images, (list, np.ndarray)):
            raise TypeError("images debe ser una lista o numpy.ndarray")
        
        if len(images) == 0:
            raise ValueError("La lista de imágenes está vacía")
        
        n_images = len(images)
        print(f"[LBP] Extrayendo características de {n_images} imágenes...")
        
        # Pre-allocar array de resultados
        features = np.zeros((n_images, self.output_dim), dtype=np.float32)
        
        # Procesar imágenes
        for idx, img in enumerate(images):
            features[idx] = self.extract(img)
            
            # Mostrar progreso cada 50 imágenes
            if (idx + 1) % 50 == 0:
                print(f"[LBP] Procesadas {idx + 1}/{n_images} imágenes")
        
        print(f"[LBP] Extracción completada.")
        return features
    
    def _get_spatial_histograms(self, lbp_image: np.ndarray) -> np.ndarray:
        """
        Calcula histogramas espaciales de una imagen LBP de forma vectorizada.
        
        Args:
            lbp_image (np.ndarray): Imagen LBP.
        
        Returns:
            np.ndarray: Vector concatenado de histogramas por celda.
        """
        h, w = lbp_image.shape
        grid_h, grid_w = self.grid_size
        
        # Calcular tamaño de cada celda
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # Pre-allocar array de histogramas
        n_cells = grid_h * grid_w
        histograms = np.zeros((n_cells, self.n_bins), dtype=np.float32)
        
        cell_idx = 0
        for i in range(grid_h):
            for j in range(grid_w):
                # Extraer región
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_h - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_w - 1 else w
                
                cell = lbp_image[y_start:y_end, x_start:x_end]
                
                # Calcular histograma con numpy (más rápido)
                hist, _ = np.histogram(cell.ravel(), bins=self.n_bins, range=(0, self.n_bins))
                
                # Normalizar histograma
                hist_sum = hist.sum()
                if hist_sum > 0:
                    histograms[cell_idx] = hist / hist_sum
                
                cell_idx += 1
        
        # Concatenar histogramas
        return histograms.ravel()