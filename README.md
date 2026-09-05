# Predicción de propensión a nuevo producto (cross-sell) con Azure MLOps

Modelo que predice qué clientes bancarios tienen mayor probabilidad de sumar un nuevo producto (tarjeta, préstamo, caja de ahorro, etc.), desarrollado con metodología **KDD** y llevado a producción end-to-end con **Azure Machine Learning**.

> **Nota sobre el target:** el análisis exploratorio inicial mostró que los clientes con `Target=1` tienen más productos activos, más actividad transaccional y contrataron su último producto más recientemente que los de `Target=0` — el patrón opuesto al de una fuga. Esto descarta la hipótesis inicial de *churn* y confirma que `Target` mide **propensión de compra / engagement**, no abandono. Esta validación temprana condicionó tanto el diseño de la ABT (Parte 1) como el enfoque de negocio del despliegue (Parte 2): el modelo no se usa para retener, sino para **priorizar a quién ofrecerle qué producto**.

---

## Estructura del repo

```
data/           datasets originales y procesados
notebooks/      exploración, limpieza y modelado (KDD)
src/            funciones reutilizables (limpieza, features, scoring)
azure/          assets de Azure ML: pipeline de entrenamiento, endpoint, monitoreo
models/         artefactos o referencia al Model Registry de Azure ML
docs/           métricas, decisiones tomadas y diagramas de arquitectura
```

El repo está pensado como un único flujo: lo que se explora y valida en `notebooks/` se traslada como código reutilizable a `src/`, y ese mismo código es el que corre dentro de los componentes del pipeline en `azure/`. No hay lógica duplicada entre el notebook y la productivización.

---

## Parte 1 — Modelado (metodología KDD)

Aplicada sobre un dataset de clientes bancarios (`client_id`, `Target`, productos activos, balances, transacciones, uso de canales), siguiendo las etapas clásicas de KDD/CRISP-DM: selección → preprocesamiento → transformación → data mining → interpretación.

### 1. Objetivo y universo (Business Understanding)
Predecir la probabilidad de que un cliente adopte un **nuevo** producto en el corto plazo, en base a su comportamiento transaccional y tenencia actual de productos.

- **Ventana temporal:** *(completar)* — training window / lead window / prediction window definidas, ej. 6 meses de histórico + 1 mes de lead + 2 meses de predicción.
- **Filtros del universo:** *(completar)* — antigüedad mínima del cliente, exclusión de clientes con el producto objetivo ya activo, exclusión de altas/bajas dentro de la ventana, etc.
- **Balance del target:** proporción de `Target=1` vs `Target=0` y estrategia de balanceo si aplica (undersampling, class weights).

### 2. Limpieza y calidad de datos
Nulos (imputación según tipo de variable — cero vs mediana vs categoría "sin dato"), duplicados, outliers (percentiles / 3-sigma), tratamiento de variables categóricas de alta cardinalidad (encoding basado en target vs one-hot).

### 3. ABT (Analytical Base Table)
Construcción de la tabla analítica a nivel `client_id`, combinando:
- **Identity features:** variables tomadas directamente de la base (edad, antigüedad, cantidad de productos activos).
- **Transform features:** variables derivadas con lógica de negocio (grupos etarios, flags binarios, ratios).
- **Aggregate features:** estadísticos por ventanas de tiempo (mínimo, promedio, máximo, mediana, suma, variación entre primer y último mes) sobre transacciones y balances.

### 4. Selección de variables
Eliminación de features sin varianza o con valores únicos, matriz de correlación, reducción de dimensionalidad (PCA) y selección basada en importancia de un modelo genérico (Random Forest / LightGBM) como primer filtro.

### 5. Algoritmo
*(completar: regresión logística, árbol, Random Forest, LightGBM, etc. y justificación — interpretabilidad vs performance, tiempo de entrenamiento, necesidad de explicabilidad para negocio)*

### 6. Validación
Train / test / validación, k-fold (o stratified k-fold dado el desbalance del target), búsqueda de hiperparámetros (Grid Search / Randomized Search).

### 7. Evaluación
Matriz de confusión con threshold ajustado (no el 0.5 por defecto), AUC-ROC, KS, análisis por deciles (lift), y comparación de performance entre train y test para descartar overfitting. Se documenta también un chequeo de **estabilidad temporal** de las variables principales, para anticipar necesidad de recalibración.

📓 Ver `notebooks/01_objetivo_y_universo.ipynb`, `02_limpieza_y_abt.ipynb`, `03_modelado_y_evaluacion.ipynb`

---

## Parte 2 — Productivización con Azure MLOps

El modelo entrenado en la Parte 1 se lleva a producción para alimentar, por ejemplo, campañas de marketing dirigidas a los clientes con mayor score de propensión.

### Arquitectura general

```
data/ (raw)
   │
   ▼
Azure ML Datastore + Data Asset  ──────────────►  versionado y trazabilidad del dataset
   │
   ▼
Azure ML Pipeline (entrenamiento)
   ├─ componente: preprocesamiento / ABT   (reutiliza src/)
   ├─ componente: entrenamiento + tuning
   └─ componente: evaluación y registro condicional
   │
   ▼
Model Registry (Azure ML)  ─────────────────────►  versión, métricas y linaje del modelo
   │
   ▼
Managed Online Endpoint  ───────────────────────►  scoring en tiempo real / batch
   │
   ▼
Monitoreo (data drift + performance)  ──────────►  alertas y disparo de reentrenamiento
```

### 1. Registro de dataset y experimento
- Carga del dataset procesado como **Data Asset** versionado en Azure ML (`azureml:cross_sell_abt:1`).
- Experimento de entrenamiento trackeado con **MLflow** (nativo en Azure ML): parámetros, métricas (AUC, KS, lift por decil) y artefactos de cada corrida.

### 2. Pipeline de entrenamiento
Pipeline declarativo (YAML + Azure ML SDK v2) que reutiliza el mismo código de `src/` usado en los notebooks, evitando duplicar lógica entre exploración y producción:
- Componente de preprocesamiento / armado de ABT.
- Componente de entrenamiento y búsqueda de hiperparámetros.
- Componente de evaluación, con **gate de calidad**: el modelo solo se registra si supera un umbral mínimo de AUC/KS respecto al modelo actualmente en producción.

### 3. Registro del modelo
Registro versionado en el **Model Registry** de Azure ML, con metadata asociada: métricas de evaluación, features utilizadas, ventana temporal de entrenamiento y referencia al experimento/commit que lo generó.

### 4. Despliegue como endpoint de inferencia
- **Managed Online Endpoint** para scoring on-demand (batch scoring diario/semanal contra la cartera de clientes para nutrir campañas).
- Script de scoring (`azure/endpoint/`) que aplica el mismo pipeline de features que en entrenamiento, para evitar *training-serving skew*.

### 5. Monitoreo de drift y degradación del modelo
- **Data drift** sobre las variables de entrada (comparando distribución de scoring vs. training).
- **Degradación de performance** en el tiempo: seguimiento de AUC/KS sobre cohortes con label ya conocido, y de la estabilidad del %target por decil (ver criterios de recalibración de la Parte 1).
- Criterios de recalibración: caída sostenida de performance, features que pierden importancia, cambios en la definición de negocio del producto ofrecido.

📁 Ver `azure/pipeline/`, `azure/endpoint/`

---

## Resultados
*(completar: AUC final, KS, lift en el primer decil, tamaño de la campaña recomendada, comparación train vs test)*

## Próximos pasos
*(completar: automatización de reentrenamiento periódico, incorporación de nuevas fuentes de datos, A/B testing de la campaña generada con el modelo)*
