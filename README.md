# Predicción y gestión de riesgo de baja de clientes con Azure MLOps

Modelo que predice qué clientes están en riesgo de dar de baja productos bancarios (tarjetas, cuentas, seguros, préstamos), desarrollado con metodología KDD y llevado a producción con Azure Machine Learning.

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

- **Objetivo y universo:** _(completar: qué es Target, ventana de tiempo, filtros aplicados)_
- **Limpieza y calidad de datos:** nulos, duplicados, outliers, tratamiento de variables categóricas
- **ABT (Analytical Base Table):** variables derivadas (mínimo, promedio, máximo, variación por ventanas de tiempo)
- **Selección de variables:** correlación, reducción de dimensionalidad, componentes principales
- **Algoritmo:** _(completar: regresión, árbol, Random Forest, LightGBM, etc. y por qué)_
- **Validación:** train/test/validación, cross-validation, búsqueda de hiperparámetros
- **Evaluación:** AUC, matriz de confusión, KS, análisis por deciles, estabilidad en el tiempo

📓 Ver `notebooks/01_objetivo_y_universo.ipynb`, `02_limpieza_y_abt.ipynb`, `03_modelado_y_evaluacion.ipynb`

## Parte 2 — Productivización con Azure MLOps

El modelo entrenado en la Parte 1 se lleva a producción:

- Registro de dataset y experimento en Azure ML
- Pipeline de entrenamiento (reutiliza el preprocesamiento del notebook)
- Registro del modelo en el Model Registry
- Despliegue como endpoint de inferencia
- Diseño de monitoreo de drift y degradación del modelo

📁 Ver `azure/pipeline/`, `azure/endpoint/`

## Resultados



## Próximos pasos

