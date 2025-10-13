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
```

Estructura:
- `configs/`: YAMLs de configuración (datos, validación, modelos, evaluación)
- `data/`: datos crudos, intermedios y procesados
- `src/`: código importable
- `scripts/`: puntos de entrada (CLI)
- `experiments/`: artefactos de ejecuciones (modelos, métricas, figuras)
- `notebooks/`: EDA y prototipos 

Licencia: MIT
