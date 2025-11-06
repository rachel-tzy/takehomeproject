from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import json
import logging

import click
import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


from .config import DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    name: str
    version: str
    run_id: str
    source: str
    params: Dict[str, str]
    metrics: Dict[str, float]


def _load_mlflow_model(
    experiment_name: str,
    name: Optional[str],
    version: Optional[int],
    run_id: Optional[str],
) -> Tuple[object, ModelMetadata]:
    """Resolve and load a model from MLflow using experiment context."""
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise click.ClickException(
            f"Experiment '{experiment_name}' was not found. Check the MLflow tracking server."
        )

    resolved_run_id = run_id
    resolved_model_uri: Optional[str] = None
    metadata: Optional[ModelMetadata] = None

    def _validate_run(target_run_id: str):
        run = client.get_run(target_run_id)
        if run.info.experiment_id != experiment.experiment_id:
            raise click.ClickException(
                f"Run '{target_run_id}' does not belong to experiment '{experiment_name}'."
            )
        return run

    if name and version is not None:
        model_version = client.get_model_version(name, str(version))
        resolved_run_id = model_version.run_id
        run = _validate_run(resolved_run_id)
        resolved_model_uri = f"models:/{model_version.name}/{model_version.version}"
        metadata = ModelMetadata(
            name=model_version.name,
            version=str(model_version.version),
            run_id=model_version.run_id,
            source=model_version.source,
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
        )
    else:
        if resolved_run_id is None:
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if not runs:
                raise click.ClickException(
                    f"No finished runs found for experiment '{experiment_name}'."
                )
            run = runs[0]
        else:
            run = _validate_run(resolved_run_id)

        resolved_run_id = run.info.run_id
        versions = client.search_model_versions(f"run_id = '{resolved_run_id}'")
        if versions:
            latest = max(versions, key=lambda mv: int(mv.version))
            resolved_model_uri = f"models:/{latest.name}/{latest.version}"
            metadata = ModelMetadata(
                name=latest.name,
                version=str(latest.version),
                run_id=resolved_run_id,
                source=latest.source,
                params=dict(run.data.params),
                metrics=dict(run.data.metrics),
            )
        else:
            run_name = (
                run.data.tags.get("mlflow.runName")
                or run.info.run_name
                or f"run-{resolved_run_id}"
            )
            model_name = run_name.replace(" ", "_")
            resolved_model_uri = f"runs:/{resolved_run_id}/model_artifacts"
            metadata = ModelMetadata(
                name=model_name,
                version="run",
                run_id=resolved_run_id,
                source=resolved_model_uri,
                params=dict(run.data.params),
                metrics=dict(run.data.metrics),
            )

    if metadata is None or resolved_model_uri is None:
        raise click.ClickException("Failed to resolve MLflow model URI.")

    model = mlflow.sklearn.load_model(resolved_model_uri)
    logger.info(
        "Loaded MLflow model name=%s version=%s run_id=%s",
        metadata.name,
        metadata.version,
        metadata.run_id,
    )
    return model, metadata


def _prepare_features(
    model, data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Split features and target with consistent preprocessing."""
    target_col = DEFAULTS.RESP_COL
    if target_col not in data.columns:
        raise click.ClickException(f"Expected target column '{target_col}' in data.")

    X = data.drop(columns=getattr(DEFAULTS, "COLUMNS_TO_DROP", []), errors="ignore")[
        model.feature_names_in_
    ]
    y = data[target_col].astype(int)
    weight_col = getattr(DEFAULTS, "WEIGHT_COL", None)
    weights = None
    if weight_col and weight_col in data.columns:
        weights = pd.to_numeric(data[weight_col], errors="coerce").fillna(1.0)
    return X, y, weights


def _compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, sample_weight: Optional[np.ndarray]
) -> Dict[str, float]:
    """Return the requested metrics given true labels and positive class probabilities."""
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds, sample_weight=sample_weight)),
        "roc_auc": float(roc_auc_score(y_true, proba, sample_weight=sample_weight)),
        "recall": float(recall_score(y_true, preds, sample_weight=sample_weight)),
        "precision": float(precision_score(y_true, preds, sample_weight=sample_weight)),
        "f1": float(f1_score(y_true, preds, sample_weight=sample_weight)),
    }


def _plot_roc_curve(
    train: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    holdout: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    save_path: Path,
) -> None:
    """Plot ROC curves for train and holdout splits."""
    plt.figure(figsize=(8, 6))
    for split_name, (y_true, proba, weights) in {
        "Train": train,
        "Holdout": holdout,
    }.items():
        fpr, tpr, _ = roc_curve(y_true, proba, sample_weight=weights)
        auc_score = roc_auc_score(y_true, proba, sample_weight=weights)
        plt.plot(fpr, tpr, label=f"{split_name} (AUC={auc_score:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _extract_feature_importance(
    model: object, feature_names: Iterable[str]
) -> pd.DataFrame:
    """Return feature importances sorted descending."""
    importance: Dict[str, float] = {}

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        if scores:
            importance = scores
    if not importance and hasattr(model, "feature_importances_"):
        importance = {
            str(name): float(score)
            for name, score in zip(feature_names, model.feature_importances_)
        }

    if not importance:
        raise click.ClickException(
            "Model does not expose feature importances compatible with this evaluator."
        )

    name_to_score = {str(k): float(v) for k, v in importance.items()}

    mapped: Dict[str, float] = {}
    feature_list = list(feature_names)
    for idx, name in enumerate(feature_list):
        key_variants = [str(name), f"f{idx}"]
        score = None
        for key in key_variants:
            if key in name_to_score:
                score = name_to_score[key]
                break
        if score is not None:
            mapped[str(name)] = score

    # Include any remaining features that were not mapped (e.g., aggregated models).
    for key, value in name_to_score.items():
        if key not in mapped:
            mapped[key] = value

    df = (
        pd.DataFrame(
            {"feature": list(mapped.keys()), "importance": list(mapped.values())}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df


def _plot_feature_importance(df: pd.DataFrame, save_path: Path) -> None:
    """Bar plot of feature importances."""
    plt.figure(figsize=(10, max(4, len(df) * 0.25)))
    plt.barh(df["feature"], df["importance"], color="#1f77b4")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_probability_distribution(
    train: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    holdout: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    save_path: Path,
) -> None:
    """Histogram of predicted probabilities for train and holdout."""
    plt.figure(figsize=(8, 6))
    plt.hist(
        train[1],
        bins=30,
        alpha=0.6,
        label="Train",
        density=True,
        weights=train[2],
    )
    plt.hist(
        holdout[1],
        bins=30,
        alpha=0.6,
        label="Holdout",
        density=True,
        weights=holdout[2],
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Prediction Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _evaluate_model(
    model,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Compute requested reports and persist them to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, w_train = _prepare_features(model, train_df)
    X_holdout, y_holdout, w_holdout = _prepare_features(model, holdout_df)

    train_proba = model.predict_proba(X_train)[:, 1]
    holdout_proba = model.predict_proba(X_holdout)[:, 1]

    metrics = {
        "train": _compute_metrics(
            y_train.to_numpy(),
            train_proba,
            None if w_train is None else w_train.to_numpy(),
        ),
        "holdout": _compute_metrics(
            y_holdout.to_numpy(),
            holdout_proba,
            None if w_holdout is None else w_holdout.to_numpy(),
        ),
    }
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "split"
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=True)

    roc_path = output_dir / "roc_curve.png"
    _plot_roc_curve(
        train=(
            y_train.to_numpy(),
            train_proba,
            None if w_train is None else w_train.to_numpy(),
        ),
        holdout=(
            y_holdout.to_numpy(),
            holdout_proba,
            None if w_holdout is None else w_holdout.to_numpy(),
        ),
        save_path=roc_path,
    )

    importance_df = _extract_feature_importance(
        model=model, feature_names=X_train.columns
    )
    importance_csv = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    _plot_feature_importance(
        importance_df, save_path=output_dir / "feature_importance.png"
    )

    prob_dist_path = output_dir / "probability_distribution.png"
    _plot_probability_distribution(
        train=(
            y_train.to_numpy(),
            train_proba,
            None if w_train is None else w_train.to_numpy(),
        ),
        holdout=(
            y_holdout.to_numpy(),
            holdout_proba,
            None if w_holdout is None else w_holdout.to_numpy(),
        ),
        save_path=prob_dist_path,
    )

    # Persist metadata for reproducibility
    metadata = {
        "metrics_path": str(metrics_path),
        "roc_curve": str(roc_path),
        "feature_importance_csv": str(importance_csv),
        "plots": [
            str(roc_path),
            str(output_dir / "feature_importance.png"),
            str(prob_dist_path),
        ],
    }
    with open(output_dir / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


@click.command()
@click.option(
    "--train-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=PROJECT_DIR / Path("assets/processed/train.parquet"),
    show_default=True,
)
@click.option(
    "--holdout-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=PROJECT_DIR / Path("assets/processed/test.parquet"),
    show_default=True,
)
@click.option(
    "--results-path",
    type=click.Path(path_type=Path, dir_okay=True),
    default=PROJECT_DIR / Path("results/evaluation"),
    show_default=True,
)
@click.option(
    "--experiment-name",
    type=str,
    default="takehome-xgboost",
    show_default=True,
)
@click.option("--model-name", type=str, default=None, show_default=True)
@click.option("--model-version", type=int, default=None, show_default=True)
@click.option("--run-id", type=str, default=None, show_default=True)
def evaluate(
    train_path: Path,
    holdout_path: Path,
    results_path: Path,
    experiment_name: str,
    model_name: Optional[str],
    model_version: Optional[int],
    run_id: Optional[str],
) -> None:
    """Load a model from MLflow and generate evaluation artifacts."""
    train_df = pd.read_parquet(train_path)
    holdout_df = pd.read_parquet(holdout_path)

    model, metadata = _load_mlflow_model(
        experiment_name=experiment_name,
        name=model_name,
        version=model_version,
        run_id=run_id,
    )
    print(model)

    output_dir = results_path / metadata.name / f"v{metadata.version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "mlflow_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.__dict__, f, indent=4)

    _evaluate_model(model, train_df, holdout_df, output_dir)
    click.echo(f"Evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    evaluate()
