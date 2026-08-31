# Predicción de propensión a nuevo producto (cross-sell) con Azure MLOps

Modelo que predice qué clientes bancarios tienen mayor probabilidad de sumar un nuevo producto (tarjeta, préstamo, caja de ahorro, etc.), desarrollado con metodología KDD y llevado a producción con Azure Machine Learning.

> **Nota sobre el target:** el análisis exploratorio inicial mostró que los clientes con `Target=1` tienen más productos activos, más actividad transaccional y contrataron su último producto más recientemente que los de `Target=0` — el patrón opuesto al de una fuga. Esto descarta la hipótesis inicial de churn y apunta a que `Target` mide propensión de compra / engagement, no abandono.

## Estructura del repo

```
data/           datasets originales y procesados
notebooks/      exploración, limpieza y modelado
src/            funciones reutilizables (limpieza, features)
azure/          pipeline de entrenamiento y endpoint de scoring
models/         artefactos o referencia al model registry
docs/           métricas y decisiones tomadas
```

## Parte 1 — Modelado

Metodología KDD aplicada sobre un dataset de clientes bancarios (`client_id`, `Target`, productos activos, balances, transacciones, uso de canales).

- **Objetivo y universo:** predecir la probabilidad de que un cliente adopte un nuevo producto en el corto plazo, en base a su comportamiento transaccional y tenencia actual de productos. _(completar: ventana de tiempo exacta y filtros aplicados al universo, ej. antigüedad mínima)_
- **Limpieza y calidad de datos:** nulos, duplicados, outliers, tratamiento de variables categóricas
- **ABT (Analytical Base Table):** variables derivadas (mínimo, promedio, máximo, variación por ventanas de tiempo)
- **Selección de variables:** correlación, reducción de dimensionalidad, componentes principales
- **Algoritmo:** _(completar: regresión, árbol, Random Forest, LightGBM, etc. y por qué)_
- **Validación:** train/test/validación, cross-validation, búsqueda de hiperparámetros
- **Evaluación:** AUC, matriz de confusión, KS, análisis por deciles, estabilidad en el tiempo

📓 Ver `notebooks/01_objetivo_y_universo.ipynb`, `02_limpieza_y_abt.ipynb`, `03_modelado_y_evaluacion.ipynb`

## Parte 2 — Productivización con Azure MLOps

El modelo de propensión entrenado en la Parte 1 se lleva a producción (por ejemplo, para alimentar campañas de marketing dirigidas a los clientes con mayor score):

- Registro de dataset y experimento en Azure ML
- Pipeline de entrenamiento (reutiliza el preprocesamiento del notebook)
- Registro del modelo en el Model Registry
- Despliegue como endpoint de inferencia
- Diseño de monitoreo de drift y degradación del modelo

📁 Ver `azure/pipeline/`, `azure/endpoint/`

## Resultados



## Próximos pasos



