# Deep Music Generation with Recurrent Neural Networks

## Descripción del problema

Este proyecto aborda la **generación automática de música** utilizando Redes Neuronales Recurrentes (RNNs). El objetivo es entrenar un modelo que aprenda patrones temporales a partir de partituras musicales representadas en notación ABC y sea capaz de generar nuevas secuencias musicales coherentes.

El trabajo se enmarca en el laboratorio del curso **MIT 6.S191 — Introduction to Deep Learning**.

## Dataset

- **Fuente:** Canciones folk irlandesas en notación ABC (proporcionadas por MIT 6.S191)
- **Tamaño:** 817 canciones, 200,679 caracteres en total
- **Vocabulario:** 83 caracteres únicos
- **Longitud promedio por canción:** 243.6 caracteres

## Decisiones técnicas

- **Framework:** PyTorch
- **Arquitectura:** LSTM character-level RNN compuesta por 3 capas:
  1. `nn.Embedding` — mapea cada carácter a un vector denso de 256 dimensiones
  2. `nn.LSTM` — 2 capas apiladas con hidden size de 1024 y dropout de 0.3
  3. `nn.Linear` — proyecta las salidas del LSTM a logits sobre el vocabulario
- **Justificación:** Se eligió LSTM por su capacidad de capturar dependencias temporales entre caracteres consecutivos, manteniendo un hidden state que actúa como memoria del modelo.
- **Optimizador:** Adam (learning rate 5e-3) con gradient clipping (norma máxima 5.0)
- **Función de pérdida:** Cross-Entropy Loss
- **Tracking de experimentos:** Comet ML
- **Entorno:** Google Colab con GPU

## Metodología

### Preprocesamiento
- Unión de todas las canciones en un corpus de texto continuo
- Creación de mapeos carácter-a-entero (`char2idx` / `idx2char`) para vectorización
- Generación de batches de entrenamiento: secuencias de longitud fija donde el target es el siguiente carácter desplazado un paso

### Entrenamiento
- 3,000 iteraciones con batch size de 8 y secuencias de 100 caracteres
- Suavizado EMA (weight=0.95) para visualización de la curva de entrenamiento

### Generación
- Proceso autoregresivo: a partir de una seed string, se generan caracteres uno a uno
- En cada paso: forward pass → logits → softmax con temperatura → muestreo con `torch.multinomial`
- Se probaron 5 temperaturas: 0.5, 0.8, 1.0, 1.2, 1.5

### Evaluación
- Perplexity sobre el training set
- Valid song rate (parsing ABC)
- N-gram diversity (n=2,3,4,5) comparando texto generado vs corpus original

## Resultados

| Métrica | Valor |
|---|---|
| Perplexity | 2.40 |
| Loss final | 0.9186 |
| Reducción de loss | 76.7% durante el entrenamiento |
| Valid song rate | 95.2% (20/21 canciones válidas) |

### Diversidad de N-gramas

| N-gram | Generado | Original | Ratio |
|---|---|---|---|
| n=2 | 0.1885 | 0.1671 | 1.13x |
| n=3 | 0.4592 | 0.3684 | 1.25x |
| n=4 | 0.6662 | 0.5043 | 1.32x |
| n=5 | 0.7952 | 0.5812 | 1.37x |

La diversidad de n-gramas superior al corpus original (ratios >1.0x) indica que el modelo generaliza en lugar de memorizar.

### Hallazgos clave
- Los LSTMs pueden aprender estructura musical (ABC) a nivel de carácter
- La temperatura controla el trade-off entre coherencia y creatividad (T=0.5 conservador, T=1.5 más creativo pero menos estable)

### Limitaciones
- Modelado a nivel carácter: sin noción explícita de notas, acordes o armonía
- Dataset limitado a un solo género musical
- Dificultad con dependencias de largo alcance

### Mejoras propuestas
- Arquitecturas Transformer para mejor modelado de largo alcance
- Tokenización musical (notas/compases/eventos)
- Datasets más grandes y diversos
- Incorporar feedback humano y métricas específicas de música
