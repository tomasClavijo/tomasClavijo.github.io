# Predicting Medical Appointment No-Shows using Machine Learning

## Descripción del problema

El proyecto aborda el **ausentismo a consultas médicas programadas (no-show)** en un sistema de salud de una ciudad de Brasil. Cuando un paciente agenda una cita y no se presenta sin cancelar, las consecuencias son:

- **Económicas:** horas de personal médico desperdiciadas e infraestructura ociosa.
- **Operativas:** turnos vacíos que podrían haber sido reasignados, aumentando tiempos de espera.
- **Sanitarias:** el paciente posterga diagnóstico o tratamiento, agravando potencialmente su condición.

El objetivo es construir un **modelo de clasificación binaria** que, al momento de una reserva, prediga si el paciente faltará o no, permitiendo tomar acciones preventivas focalizadas (llamadas de confirmación, sobre-reservas controladas, reasignación dinámica de turnos).

## Dataset

- **Fuente:** Datos de consultas médicas de Vitória, ES, Brasil
- **Tamaño:** ~110,527 registros
- **Período:** Abril–Junio 2016
- **Desbalance:** 79.8% asistieron / 20.2% faltaron (ratio ~4:1)
- **Variables:** 14 originales, incluyendo datos demográficos (género, edad, barrio), condiciones médicas (hipertensión, diabetes, alcoholismo, discapacidad), programa social (Bolsa Família), envío de SMS y fechas de reserva/consulta

## Decisiones técnicas

- **Librerías:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, LightGBM, SHAP
- **Modelos evaluados:** Regresión Logística (baseline), Random Forest, XGBoost y LightGBM
- **Métrica principal:** Recall (no detectar una inasistencia es más costoso que una falsa alarma)
- **Métricas complementarias:** Precision, F1-Score, AUC-ROC
- **Optimización:** RandomizedSearchCV con 40 iteraciones y 3 folds
- **Interpretabilidad:** SHAP values (TreeExplainer)
- **Entorno:** Google Colab

## Metodología

### Exploración y limpieza
- Sin valores nulos en el dataset
- Eliminados 6 registros anómalos (1 con edad negativa, 5 con fechas inconsistentes)
- Eliminados 3 barrios con menos de 10 registros

### Feature Engineering (9 variables creadas)
- `DiasAnticipacion` — días entre reserva y consulta
- `DiaSemana` — día de la semana de la consulta
- `MesConsulta` — mes de la consulta
- `HoraReserva` — hora de la reserva
- `RangoEtario` — grupo etario
- `CitasPrevias` — conteo acumulado de citas del paciente
- `Gender_encoded` — género codificado
- `Barrio_encoded` — target encoding con tasa histórica de no-show del barrio
- `Target` — variable objetivo codificada

### Entrenamiento y evaluación
- Split 80/20 estratificado
- Validación cruzada estratificada de 5 folds
- Análisis de umbral de decisión (curva Precision-Recall)
- Selección de features por importancia (top 6 vs 14)

## Resultados

### Comparación de modelos (validación cruzada 5-Fold)

| Modelo | F1 | AUC-ROC | Recall | Precision |
|---|---|---|---|---|
| Regresión Logística | 0.035 ± 0.004 | 0.668 ± 0.005 | 0.019 ± 0.002 | 0.340 ± 0.037 |
| Random Forest | 0.194 ± 0.001 | 0.740 ± 0.004 | 0.120 ± 0.001 | 0.515 ± 0.011 |
| XGBoost Optimizado | 0.456 ± 0.002 | 0.745 ± 0.003 | 0.816 ± 0.008 | 0.316 ± 0.001 |
| **LightGBM Optimizado** | **0.456 ± 0.003** | **0.746 ± 0.002** | **0.810 ± 0.008** | **0.317 ± 0.002** |

### Modelo seleccionado: LightGBM Optimizado

- De cada 100 pacientes que faltan, el modelo detecta 81.
- De cada 100 alertas, 32 son correctas y 68 son falsas alarmas (costo bajo: solo una llamada preventiva).
- Solo 19 de cada 100 inasistencias pasan sin ser detectadas.

### Top 6 features más importantes
1. **DiasAnticipacion** — predictor dominante (mismo día: 4.7% no-show vs >30 días: 33%)
2. **Edad** — jóvenes faltan más (adolescentes: 26.1%, adultos jóvenes: 23.8%)
3. **Barrio_encoded** — variabilidad geográfica (15.8% a 26.3%)
4. **HoraReserva**
5. **CitasPrevias**
6. **DiaSemana**

### Hallazgo notable: SMS
El SMS recordatorio no redujo la inasistencia (27.6% con SMS vs 16.7% sin SMS), debido a un sesgo de selección: los SMS se enviaron preferentemente a pacientes con citas más lejanas (correlación 0.40 entre Recibió SMS y DiasAnticipacion).

### Limitaciones
- Precision baja (0.32): muchas falsas alarmas
- Target encoding del barrio tiene leve data leakage
- Ventana temporal limitada (2 meses, no captura estacionalidad anual)
- AUC-ROC de 0.746: capacidad discriminativa moderada
