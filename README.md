# Income Classification Take-Home Project

This repository contains code to preprocess, model, and evaluate the income-classification task described in the take-home instructions. A segmentation workflow is being built in parallel; its open pieces are noted below.

## Project Structure

- `src/preprocessing.py` �?CLI for generating processed datasets (classification & segmentation variants).
- `src/train_classification.py` �?CLI to train the XGBoost classifier with hyperparameter search and MLflow logging.
- `src/evaluate.py` �?CLI to load an MLflow model and produce metrics/plots.
- `src/segmentation.py` �?rule-based segmentation CLI with weighted profiling.
- `assets/raw/` �?raw census files (`census-bureau.data`, `.columns`).
- `assets/processed/` �?processed data.
- `mlruns/` �?local MLflow tracking data.

## Environment Setup

1. **Install Poetry (once per machine)**
   ```
   pipx install poetry
   ```
   In this project, I used **poetry** to manage dependencies and environment. If you have trouble to run the above command, please refer the [offcial installation instruction](https://python-poetry.org/docs/#installation). 
   
2. **Install dependencies**
   ```bash
   poetry install
   ```
   This uses the locked versions in `poetry.lock`.

## Preprocessing Datasets

The preprocessing CLI command enable preprocess data for classification model and segmentation model. 

- **Classification**

You can run ```poetry run preprocess classification --help``` to see available arguments.

To preprocess the data using default parameters: 
  ```bash
  poetry run preprocess classification 
  ```

## Training the Classification Model

Run the training CLI against the processed classification data:

```bash
poetry run train-classification --help
```

Key options:

```
--train-path FILE               Path to the processed training dataset (parquet). [required]
-experiment-name TEXT           MLflow experiment name. [required]
--reduce-method [minfo|extra-trees|all]
                                Feature selector: mutual information (`minfo`), ExtraTrees (`extra-trees`), or all features.
-t, --tune-method [hyperopt|manual]
                                Hyperopt tuning or manual parameter set. [default: hyperopt]
-e, --early-stopping [auc|aucpr|none]
                                Early stopping metric. [default: auc]
--undersample [none|random|near-miss]
                                Optional class undersampling strategy. [default: none]
--tune-loss [roc_auc|average_precision|f1_weighted|neg_log_loss|partial_auc]
                                Objective for Hyperopt. [default: roc_auc]
-k, --n-features INTEGER        Number of features to select (for selectors). [default: 20]
-n, --n-samples INTEGER         Subsample rows for quick runs.
--n-trials INTEGER              Hyperopt evaluations. [default: 50]
-j, --n-jobs INTEGER            Parallel jobs. [default: 1]
-r, --random-seed INTEGER       Random seed. [default: 42]
```

All runs log parameters, metrics, and selected features to MLflow (`mlruns/`).

## Evaluating a Model

Evaluate any trained model registered/logged in MLflow:

```bash
poetry run evaluate \
  --experiment-name takehome-xgboost \
  --train-path assets/processed_classification/train.parquet \
  --holdout-path assets/processed_classification/test.parquet \
  --results-path results/evaluation \
  --model-name <registered-model-name> \
  --model-version <int>
```

Outputs include:

- `metrics.csv` with accuracy, ROC AUC, precision, recall, F1 (weighted by sample weights).
- `roc_curve.png`, `feature_importance.csv/png`, `probability_distribution.png`.
- MLflow metadata & artifacts stored under `<results-path>/<model-name>/v<version>/`.

## Segmentation Workflow

Run the rule-based segmentation CLI against the combined processed dataset:

```bash
poetry run segment \
  --data-path assets/processed/data_unmapped.parquet \
  --output-dir results/segmentation_rules \
  --weight-col weight
```

Outputs include:

- `rule_segments.parquet`: row-level segment assignments.
- `segment_profiles.parquet`: weighted numeric/categorical summaries.
- `SUMMARY.json`: quick reference to the inputs, outputs, and rule names.


## Notebooks
You can check `sandbox/EDA.ipynb` to get the code where the plots in the report are generated.

All notebooks assume paths relative to the repo root. When launching from `sandbox/`, ensure the project root is on `sys.path`. Example:

```python
import sys
from pathlib import Path
project_root = Path.cwd()
if not (project_root / "<package_root>").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
```

