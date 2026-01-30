import numpy as np
import cv2


class LBPExtractor:
    """
    Extractor LBP (Local Binary Patterns).
    
    Extrae características de textura basadas en patrones binarios locales,
    ampliamente utilizados en reconocimiento de texturas y rostros.
    
    Attributes:
        radius (int): Radio del operador LBP.
        n_points (int): Número de puntos de muestreo alrededor del pixel central.
        method (str): Método de cálculo ('default', 'ror', 'uniform', 'nri_uniform', 'var').
        grid_size (tuple): Tamaño de la grilla para histogramas espaciales.
        output_dim (int): Dimensionalidad del vector de salida.
    """
    
    def __init__(self, radius: int = 1, n_points: int = 8, method: str = 'uniform',
                 grid_size: tuple = (8, 8), output_dim: int = 256):
        """
        Inicializa el extractor LBP.
        
        Args:
            radius (int): Radio del operador LBP. Default: 1.
            n_points (int): Número de puntos de muestreo. Default: 8.
            method (str): Método de cálculo ('uniform', 'default', 'ror', etc). Default: 'uniform'.
            grid_size (tuple): Tamaño de la grilla (filas, columnas). Default: (8, 8).
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
        else:
            # Para método default: 2^n_points bins
            self.n_bins = 2 ** n_points
        
        # Inicializar tabla de lookup para patrones uniformes
        if method == 'uniform':
            self.uniform_map = self._uniform_pattern_map()
    
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
        
        # Calcular imagen LBP
        lbp_image = self._compute_lbp(gray)
        
        # Calcular histogramas espaciales
        feature_vector = self._get_spatial_histograms(lbp_image)
        
        # Normalizar
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / (norm + 1e-6)
        
        # Redimensionar a output_dim
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
        
        # Extraer características de cada imagen
        features = []
        for img in images:
            feat = self.extract(img)
            features.append(feat)
        
        # Apilar resultados en una matriz
        return np.array(features, dtype=np.float32)
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula la imagen LBP.
        
        Args:
            image (np.ndarray): Imagen en escala de grises.
        
        Returns:
            np.ndarray: Imagen LBP con códigos de patrones.
        """
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint16)
        
        # Para cada pixel (excepto bordes)
        for y in range(self.radius, h - self.radius):
            for x in range(self.radius, w - self.radius):
                center = image[y, x]
                
                # Obtener vecinos
                neighbors = self._get_neighbors(y, x, image)
                
                # Generar código binario
                binary_code = 0
                for i, neighbor_val in enumerate(neighbors):
                    if neighbor_val >= center:
                        binary_code |= (1 << i)
                
                # Aplicar método
                if self.method == 'uniform':
                    # Mapear a patrón uniforme
                    lbp[y, x] = self.uniform_map.get(binary_code, self.n_bins - 1)
                else:
                    lbp[y, x] = binary_code
        
        return lbp
    
    def _get_spatial_histograms(self, lbp_image: np.ndarray) -> np.ndarray:
        """
        Calcula histogramas espaciales de una imagen LBP.
        
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
        
        histograms = []
        
        # Para cada celda
        for i in range(grid_h):
            for j in range(grid_w):
                # Extraer región
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_h - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_w - 1 else w
                
                cell = lbp_image[y_start:y_end, x_start:x_end]
                
                # Calcular histograma
                hist, _ = np.histogram(cell.flatten(), bins=self.n_bins, 
                                       range=(0, self.n_bins))
                
                # Normalizar histograma
                hist = hist.astype(np.float32)
                hist_sum = hist.sum()
                if hist_sum > 0:
                    hist = hist / hist_sum
                
                histograms.append(hist)
        
        # Concatenar histogramas
        return np.concatenate(histograms)
    
    def _get_neighbors(self, y: int, x: int, image: np.ndarray) -> list:
        """
        Obtiene los píxeles vecinos en coordenadas circulares.
        
        Args:
            y (int): Coordenada Y del píxel central.
            x (int): Coordenada X del píxel central.
            image (np.ndarray): Imagen.
        
        Returns:
            list: Lista de valores de píxeles vecinos.
        """
        neighbors = []
        
        for i in range(self.n_points):
            # Calcular ángulo
            angle = 2 * np.pi * i / self.n_points
            
            # Calcular coordenadas circulares
            neighbor_x = x + self.radius * np.cos(angle)
            neighbor_y = y - self.radius * np.sin(angle)
            
            # Interpolar valor (bilinear)
            neighbor_val = self._bilinear_interpolation(image, neighbor_y, neighbor_x)
            neighbors.append(neighbor_val)
        
        return neighbors
    
    def _bilinear_interpolation(self, image: np.ndarray, y: float, x: float) -> float:
        """
        Interpolación bilineal para obtener valor en coordenadas no enteras.
        
        Args:
            image: Imagen.
            y: Coordenada Y (puede ser float).
            x: Coordenada X (puede ser float).
        
        Returns:
            Valor interpolado.
        """
        h, w = image.shape
        
        # Coordenadas enteras
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)
        
        # Pesos
        wx = x - x0
        wy = y - y0
        
        # Interpolación
        val = (1 - wx) * (1 - wy) * image[y0, x0] + \
              wx * (1 - wy) * image[y0, x1] + \
              (1 - wx) * wy * image[y1, x0] + \
              wx * wy * image[y1, x1]
        
        return val
    
    def _uniform_pattern_map(self) -> dict:
        """
        Genera tabla de lookup para patrones uniformes.
        
        Returns:
            dict: Mapeo de patrones a índices de bin.
        """
        uniform_map = {}
        bin_index = 0
        
        # Para cada posible código binario
        for code in range(2 ** self.n_points):
            # Contar transiciones 0->1
            transitions = 0
            binary_str = format(code, f'0{self.n_points}b')
            
            for i in range(self.n_points):
                if binary_str[i] != binary_str[(i + 1) % self.n_points]:
                    transitions += 1
            
            # Si tiene máximo 2 transiciones, es uniforme
            if transitions <= 2:
                uniform_map[code] = bin_index
                bin_index += 1
        
        return uniform_map