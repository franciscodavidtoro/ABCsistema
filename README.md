# Sistema de Reidentificación de Personas

## Descripción

Sistema integrado de reconocimiento y reidentificación de personas basado en visión por computador. El sistema combina técnicas de detección facial, análisis de apariencia corporal y clasificación mediante SVM para identificar personas en videos.

## Estructura del Proyecto

```
ABCsistema/
├── data/                   # Directorio para datasets
│   ├── dataset/            # Videos originales organizados por persona
│   ├── datasetPros/        # Imágenes procesadas (rostros y cuerpos)
│   ├── evaluations/        # Resultados de evaluaciones
│   └── features/           # Vectores de características extraídos
├── detection/              # Detección de rostros y cuerpos
│   ├── __init__.py
│   ├── face_detection.py   # Detector de rostros (YOLO)
│   └── body_detection.py   # Detector de cuerpos (YOLO)
├── feature_extraction/     # Extracción de características
│   ├── __init__.py
│   ├── facial_features.py  # Características faciales
│   ├── body_features.py    # Características corporales
│   ├── feature_vector.py   # Vector de características combinado
│   ├── hog.py              # Histogram of Oriented Gradients
│   ├── hsv.py              # Características de color HSV
│   └── lbp.py              # Local Binary Patterns
├── interface/              # Interfaz de línea de comandos
│   ├── __init__.py
│   └── command_handler.py
├── models/                 # Directorio para modelos entrenados
├── preprocessing/          # Preprocesamiento de videos e imágenes
│   ├── __init__.py
│   ├── frame_extraction.py     # Extracción de frames a 10 FPS
│   ├── data_augmentation.py    # Augmentation de imágenes
│   ├── preprocessing_pipeline.py # Pipeline completo
│   └── preprocessors/          # Preprocesadores base
│       ├── __init__.py
│       └── base_preprocessor.py
├── svm_classifier/         # Clasificador SVM
│   ├── __init__.py
│   ├── svm_model.py
│   ├── model_trainer.py
│   └── model_evaluator.py
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

## Características Principales

### 1. Preprocesamiento (`preprocessing/`)
- **Extracción de frames**: Extrae fotogramas de videos a 10 FPS
- **Detección automática**: Detecta rostros y cuerpos usando YOLO
- **Recorte inteligente**: Recorta y redimensiona regiones detectadas
  - Rostros: 256x256 píxeles
  - Cuerpos: 256x512 píxeles
- **Data augmentation**: Rotaciones, brillo, contraste, flip, ruido
- **Formato de salida**: PNG con nomenclatura secuencial

### 2. Detección (`detection/`)
- Detección de rostros con YOLOv8
- Detección de cuerpos completos con YOLOv8
- Recorte y organización de regiones detectadas
- Umbral de confianza configurable

### 3. Extracción de Características (`feature_extraction/`)
- **HOG**: Histogram of Oriented Gradients para forma
- **LBP**: Local Binary Patterns para textura
- **HSV**: Histogramas de color en espacio HSV
- Vectores normalizados para clasificación

### 4. Clasificación SVM (`svm_classifier/`)
- Entrenamiento de modelo SVM con características
- Validación cruzada y evaluación
- Ajuste de hiperparámetros
- Predicción con nivel de confianza

### 5. Interfaz CLI (`interface/`)
- Comando de preprocesamiento
- Entrenamiento de modelos
- Evaluación de desempeño
- Identificación de personas

## Requisitos

- Python 3.8 o superior
- CUDA (opcional, para aceleración GPU)
- Librerías especificadas en `requirements.txt`

## Instalación

1. **Clonar el repositorio**
```bash
git clone <url-repo>
cd ABCsistema
```

2. **Crear entorno virtual**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## Uso

### Pipeline de Preprocesamiento

```python
from preprocessing import create_pipeline

# Crear pipeline con configuración predeterminada
pipeline = create_pipeline(
    dataset_path="data/dataset",
    output_path="data/datasetPros"
)

# Ejecutar pipeline completo (extracción + augmentation)
stats = pipeline.run_full_pipeline(augmentation_multiplier=3)

# O ejecutar pasos individuales
pipeline.extract_frames_only()
pipeline.augment_only(augmentation_multiplier=3)

# Ver reporte
print(pipeline.generate_report())
```

### Modo Interactivo CLI
```bash
python main.py
```

### Comandos Disponibles

```bash
# Preprocesamiento
> preprocess <dataset_path> <output_path> [--fps 10] [--augmentation 3]

# Entrenar modelo SVM
> train_svm <dataset_path> <model_output_path>

# Evaluar modelo
> evaluate <model_path> <test_dataset_path>

# Identificar persona
> identify <image_path> <model_path> [--threshold 0.7]
```

## Estructura de Datos

### Dataset Original (entrada)
```
data/dataset/
├── persona1/
│   ├── front/
│   │   └── video1.mp4
│   └── back/
│       └── video2.mp4
├── persona2/
│   ├── front/
│   │   └── video1.mp4
│   └── back/
│       └── video2.mp4
└── ...
```

### Dataset Procesado (salida)
```
data/datasetPros/
├── persona1/
│   ├── face/           # Rostros (solo de videos front/)
│   │   ├── img0001.png
│   │   ├── img0002.png
│   │   └── ...
│   ├── front/          # Cuerpos frontales
│   │   ├── img0001.png
│   │   └── ...
│   └── back/           # Cuerpos traseros
│       ├── img0001.png
│       └── ...
├── persona2/
│   └── ...
└── ...
```

## Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE PREPROCESAMIENTO                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ENTRADA: Videos organizados por persona                      │
│     dataset/{persona}/{front|back}/video.mp4                     │
│                          │                                       │
│                          ▼                                       │
│  2. EXTRACCIÓN DE FRAMES (10 FPS)                               │
│     - Lee video con OpenCV                                       │
│     - Extrae 1 frame cada video_fps/10 frames                   │
│                          │                                       │
│                          ▼                                       │
│  3. DETECCIÓN (YOLO)                                            │
│     - Detecta personas en cada frame                            │
│     - Recorta rostros (solo de front/)                          │
│     - Recorta cuerpos (front/ y back/)                          │
│                          │                                       │
│                          ▼                                       │
│  4. REDIMENSIONAMIENTO                                          │
│     - Rostros: 256x256                                          │
│     - Cuerpos: 256x512                                          │
│                          │                                       │
│                          ▼                                       │
│  5. DATA AUGMENTATION (opcional)                                │
│     - Rotaciones aleatorias (±15°)                              │
│     - Ajuste de brillo (0.8-1.2)                                │
│     - Ajuste de contraste (0.8-1.2)                             │
│     - Flip horizontal                                           │
│     - Ruido gaussiano                                           │
│                          │                                       │
│                          ▼                                       │
│  6. SALIDA: Imágenes PNG organizadas                            │
│     datasetPros/{persona}/{face|front|back}/img####.png         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Modelos Utilizados

| Componente | Modelo | Descripción |
|------------|--------|-------------|
| Detección | YOLOv8n | Detección de personas (clase 0 COCO) |
| Rostros | YOLOv8n | Extrae parte superior del body detectado |
| Características | HOG, LBP, HSV | Descriptores de forma, textura y color |
| Clasificación | SVM | Kernel RBF por defecto |

## Métricas de Evaluación

- Exactitud (Accuracy)
- Precisión y Recall por clase
- F1-Score
- Matriz de Confusión
- Curva ROC y AUC

## Configuración del Pipeline

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| FPS | 10 | Frames por segundo a extraer |
| face_resolution | (256, 256) | Resolución de rostros |
| body_resolution | (256, 512) | Resolución de cuerpos |
| confidence_threshold | 0.5 | Umbral de confianza YOLO |
| augmentation_multiplier | 3 | Augmentaciones por imagen |

## Contribuciones

Se aceptan contribuciones. Por favor:
1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit cambios
4. Push a la rama
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo licencia MIT.

## Contacto

Para preguntas o sugerencias, contactar al equipo de desarrollo.

---

**Última actualización**: 29 de enero de 2026
