# ML Project Scaffold

Proyecto base para experimentos de Machine Learning con scikit-learn.

## Flujo rápido
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
pip install -r requirements.txt

# KNN for default
python scripts/prepare_data.py
python scripts/train.py --model-config configs/model_knn.yaml
python scripts/evaluate.py

# Backpropagation classifier
python scripts/train.py --model-config configs/model_backprop.yaml

# SVM classifier
python scripts/train.py --model-config configs/model_svm.yaml

# Benchmark (override configs desde CLI)
python scripts/prepare_water.py
python -m scripts.benchmark --benchmark-config configs/benchmark.yaml \
	--data-config configs/data_water.yaml \
	--model-config configs/model_svm.yaml

# Visualizar frontera del SVM (ejemplo con Iris)
python scripts/prepare_data.py
python -m scripts.plot_svm --features petal_length,petal_width \
	--model-config configs/model_svm.yaml \
	--output experiments/figures/svm_iris.png
```

Estructura:
- `configs/`: YAMLs de configuración (datos, validación, modelos, evaluación)
- `data/`: datos crudos, intermedios y procesados
- `src/`: código importable
- `scripts/`: puntos de entrada (CLI)
- `experiments/`: artefactos de ejecuciones (modelos, métricas, figuras)
- `notebooks/`: EDA y prototipos 

Modelos disponibles:
- `knn`: Clasificador k-vecinos de scikit-learn.
- `nearest_centroid`: Clasificador de centroides cercanos.
- `backprop`: Perceptrón multicapa implementado internamente con descenso por mini-batches y early stopping controlable vía `configs/model_backprop.yaml`.
- `svm`: Clasificador de Máquinas de Vectores de Soporte (`sklearn.svm.SVC`).

`scripts/benchmark.py` acepta `--data-config` y `--model-config` para evitar modificar el YAML cada vez que cambies de dataset/modelo.
`scripts/plot_svm.py` genera una gráfica 2D con la frontera de decisión, soporte e hiperplano (si el kernel es lineal) para el par de columnas que especifiques.

Licencia: MIT
