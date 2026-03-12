# Helmet Detection using Computer Vision

## Descripción del problema

Este proyecto aborda la **seguridad laboral en obras de construcción**. Según la OIT, al menos 60,000 trabajadores mueren cada año en accidentes en obras de construcción en todo el mundo. En Uruguay, se registran aproximadamente 40,000 accidentes laborales por año, de los cuales 6,000 corresponden al sector de la construcción. Un número significativo de estos accidentes se vincula al no uso del Equipo de Protección Personal (EPP), como cascos de seguridad.

El objetivo es desarrollar un sistema basado en deep learning capaz de **clasificar automáticamente si los trabajadores en obras de construcción están usando cascos de seguridad**, con potencial de integrarse con cámaras de vigilancia existentes para monitoreo y alertas en tiempo real.

## Dataset

- **Nombre:** SHEL5K (Safety HELmet dataset with 5K images)
- **Fuente:** [Kaggle — Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- **Tamaño:** 5,000 imágenes con 75,570 instancias anotadas
- **Formato de anotaciones:** Pascal VOC (XML)
- **Clases originales (6):** helmet, head, head with helmet, person with helmet, person without helmet, face
- **Agrupación binaria:** Con casco (1) / Sin casco (0)
- **Total de recortes procesados:** 19,187 (limitados a 10 por imagen)
- **Desbalance:** ~78.5% con casco / ~21.5% sin casco

## Decisiones técnicas

- **Framework:** TensorFlow / Keras
- **Librerías auxiliares:** OpenCV, NumPy, Matplotlib, scikit-learn, kagglehub
- **Modelo:** CNN personalizada diseñada desde cero (sin transfer learning)
- **Tarea:** Clasificación binaria
- **Global Average Pooling** en lugar de Flatten para reducir parámetros y riesgo de overfitting
- **Optimizador:** Adam (learning rate inicial 1e-4)
- **Función de pérdida:** Binary Cross-Entropy
- **Entorno:** Google Colab con GPU T4

## Metodología

### Preprocesamiento
1. Extracción de bounding boxes anotadas en XML (Pascal VOC)
2. Recorte de cada región de interés de la imagen original
3. Agrupación de 6 clases en 2 clases binarias
4. Redimensionamiento a 64x64 píxeles y normalización al rango [0, 1]

### Data Augmentation (solo en entrenamiento)
- Flip horizontal aleatorio
- Variación de brillo (max_delta=0.15)
- Variación de contraste (0.85–1.15)

### Split de datos (estratificado, random_state=42)
- **Train:** 13,437 muestras (70%)
- **Validación:** 2,871 muestras (15%)
- **Test:** 2,879 muestras (15%)

### Arquitectura del modelo (423,361 parámetros — 1.61 MB)

| Capa | Detalles |
|------|----------|
| Bloque 1 | Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPooling2D |
| Bloque 2 | Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPooling2D |
| Bloque 3 | Conv2D(128, 3x3) + BatchNorm + ReLU + MaxPooling2D + Dropout(0.3) |
| Bloque 4 | Conv2D(256, 3x3) + BatchNorm + ReLU + MaxPooling2D + Dropout(0.4) |
| Clasificador | GlobalAveragePooling2D + Dense(128, relu) + Dropout(0.5) + Dense(1, sigmoid) |

### Entrenamiento
- Máximo 50 epochs (entrenó las 50 completas)
- `ReduceLROnPlateau`: reduce lr si val_loss no mejora en 5 epochs (factor=0.5)
- `EarlyStopping`: patience de 10 epochs con restore_best_weights
- `ModelCheckpoint`: guarda el mejor modelo según val_auc
- Learning rate se redujo automáticamente: 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5

## Resultados

### Métricas en Test Set

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 97.95% |
| **Precision** | 99.02% |
| **Recall** | 98.36% |
| **AUC-ROC** | 0.9943 |
| Loss | 0.0639 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Sin casco | 0.94 | 0.96 | 0.95 | 618 |
| Con casco | 0.99 | 0.98 | 0.99 | 2,261 |
| **Weighted avg** | **0.98** | **0.98** | **0.98** | **2,879** |

### Observaciones
- El AUC-ROC de 0.9943 indica excelente capacidad de discriminación
- El recall de "sin casco" (0.96) significa que el 4% de personas sin protección no serían detectadas, un punto a mejorar en un contexto de seguridad laboral

### Limitaciones
- Imágenes de entrada de 64x64 px limitan la resolución para detección a distancia
- Sin transfer learning; modelos preentrenados (ResNet, EfficientNet) podrían mejorar el rendimiento
- Clasificación binaria simple; no incluye detección de objetos ni localización espacial
