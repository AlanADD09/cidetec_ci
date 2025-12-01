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

Licencia: MIT
