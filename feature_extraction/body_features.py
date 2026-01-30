"""
Módulo para la extracción de características corporales.

Extrae vectores numéricos representativos de cuerpos (vista frontal y posterior)
capturando silueta, proporciones y patrones visuales.

Soporta tres métodos de extracción:
- 'hog': Histogram of Oriented Gradients (descriptores de forma)
- 'hsv': Histogramas de color en espacio HSV
- 'lbp': Local Binary Patterns (descriptores de textura)
"""

import numpy as np
from .hog import HOGExtractor
from .hsv import HSVExtractor
from .lbp import LBPExtractor


class BodyFeatureExtractor:
    """
    Clase encargada de extraer características discriminativas de cuerpos.
    
    Attributes:
        method (str): Método de extracción ('hog', 'hsv', 'lbp').
        extractor: Instancia del extractor específico.
    """
    
    AVAILABLE_METHODS = {
        'hog': HOGExtractor,
        'hsv': HSVExtractor,
        'lbp': LBPExtractor,
    }
    
    def __init__(self, method='hog', **kwargs):
        """
        Inicializa el extractor de características corporales.
        
        Args:
            method (str): Método de extracción ('hog', 'hsv', 'lbp'). Default: 'hog'.
            **kwargs: Parámetros adicionales para el extractor específico.
        
        Raises:
            ValueError: Si el método no es válido.
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Método '{method}' no válido. Opciones disponibles: {list(self.AVAILABLE_METHODS.keys())}"
            )
        
        self.method = method
        self.extractor = self.AVAILABLE_METHODS[method](**kwargs)
    
    def extract(self, image):
        """
        Extrae características de una imagen corporal.
        
        Args:
            image (numpy.ndarray): Imagen en formato numpy array (cuerpo).
        
        Returns:
            numpy.ndarray: Vector de características.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Se espera numpy.ndarray, se recibió {type(image)}")
        
        return self.extractor.extract(image)
    
    def extract_batch(self, images):
        """
        Extrae características de un lote de imágenes corporales.
        
        Args:
            images (list): Lista de imágenes de cuerpos (numpy arrays).
        
        Returns:
            numpy.ndarray: Matriz de características de dimensión (N, feature_dim).
        """
        if not isinstance(images, (list, np.ndarray)):
            raise TypeError(f"Se espera lista o numpy.ndarray, se recibió {type(images)}")
        
        return self.extractor.extract_batch(images)
    
    def extract_from_directory(self, directory_path):
        """
        Extrae características de todas las imágenes en un directorio.
        
        Args:
            directory_path (str): Ruta del directorio con imágenes.
        
        Returns:
            tuple: (features, file_paths, labels) con matriz, rutas y etiquetas.
        """
        # TODO: Listar imágenes en directorio
        # TODO: Cargar y extraer características
        # TODO: Generar etiquetas (front/back según estructura)
        # TODO: Retornar matriz, rutas y etiquetas
        pass
    
    def extract_front_and_back(self, front_images, back_images):
        """
        Extrae características de vistas frontal y posterior.
        
        Args:
            front_images (list): Lista de imágenes de vista frontal.
            back_images (list): Lista de imágenes de vista posterior.
        
        Returns:
            dict: Diccionario con características separadas {'front': ..., 'back': ...}
        """
        # TODO: Extraer características de imágenes frontal
        # TODO: Extraer características de imágenes posterior
        # TODO: Organizar en diccionario
        # TODO: Retornar diccionario
        pass
    
    def extract_and_save(self, image_path, output_path):
        """
        Extrae características y las guarda en archivo.
        
        Args:
            image_path (str): Ruta de la imagen.
            output_path (str): Ruta donde guardar las características.
        
        Returns:
            bool: True si se guardó exitosamente.
        """
        # TODO: Cargar imagen
        # TODO: Extraer características
        # TODO: Guardar en formato apropiado
        # TODO: Retornar estado
        pass
