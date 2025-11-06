from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import warnings

import click
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import scope
from imblearn.pipeline import Pipeline
from mlflow.models import infer_signature
from mlflow.sklearn import SERIALIZATION_FORMAT_CLOUDPICKLE
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import DEFAULTS
from .selectors import get_feature_selector
from .utils import get_classifier_params, get_select_params

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
DEFAULT_TRAIN_DATA = (
    Path(__file__).resolve().parent.parent.parent / "assets/processed/train.parquet"
)


@click.command()
@click.option(
    "--train-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_TRAIN_DATA,
    required=True,
    help="Path to the processed training dataset (parquet).",
)
@click.option(
    "-experiment-name",
    type=str,
    default="takehome-xgboost",
    required=True,
    help="mlflow experiment name",
)
@click.option(
    "--reduce-method",
    type=click.Choice(
        [
            "minfo",
            "extra-trees",
            "all",
        ]
    ),
    default="extra-trees",
    show_default=True,
    help=(
        "Mutual information features selection (minfo), "
        "best features from an ExtraTreesClassifier (extra-trees), "
        "all features (all)"
    ),
)
@click.option(
    "-t",
    "--tune-method",
    "tune_method",
    type=click.Choice(["hyperopt", "manual"]),
    default="hyperopt",
    show_default=True,
    help=(
        "Whether to perform hyperparameter tuning (hyperopt) "
        "or use the manually selected parameters (manual)."
    ),
)
@click.option(
    "-e",
    "--early-stopping",
    type=click.Choice(["auc", "aucpr", "none"]),
    default="auc",
    show_default=True,
    help="Which early stopping method to use, if any.",
)
@click.option(
    "--tune-loss",
    type=click.Choice(
        [
            "roc_auc",
            "average_precision",
            "f1_weighted",
            "neg_log_loss",
            "partial_auc",
        ]
    ),
    default="roc_auc",
    show_default=True,
    help="Loss metric to use for hyperparameter tuning.",
)
@click.option(
    "-k",
    "--n-features",
    "n_features",
    type=int,
    default=20,
    show_default=True,
    help="Number of features to select",
)
@click.option(
    "-n",
    "--n-samples",
    "n_samples",
    type=int,
    default=None,
    show_default=True,
    help="Number of rows to sample (for testing)",
)
@click.option(
    "--n-trials",
    type=int,
    default=50,
    show_default=True,
    help="The number of tuning trials to perform.",
)
@click.option(
    "-j",
    "--n-jobs",
    type=int,
    default=1,
    show_default=True,
    help="The number of jobs to run in parallel.",
)
@click.option(
    "-r",
    "--random-seed",
    "seed",
    type=int,
    default=42,
    show_default=True,
    help="A random seed for reproducibility",
)
@click.pass_context
def train(
    ctx: click.Context,
    train_path: Path,
    experiment_name: str,
    reduce_method: Literal["minfo", "extra-trees", "all"],
    tune_method: Literal["hyperopt", "manual"],
    early_stopping: Literal["auc", "aucpr", "none"],
    tune_loss: Literal[
        "roc_auc",
        "average_precision",
        "f1_weighted",
        "neg_log_loss",
        "partial_auc",
    ],
    n_features: int,
    n_samples: Optional[int],
    n_trials: int,
    n_jobs: int,
    seed: Optional[int],
):
    """Train an xgboost classifier to income >50k or not."""
    _train(
        train_path=train_path,
        experiment_name=experiment_name,
        reduce_method=reduce_method,
        tune_method=tune_method,
        early_stopping=early_stopping,
        tune_loss=tune_loss,
        n_features=n_features,
        n_samples=n_samples,
        n_trials=n_trials,
        n_jobs=n_jobs,
        seed=seed,
    )


# Core training logic


def _train(
    train_path: Path,
    experiment_name: str,
    reduce_method: Literal["minfo", "extra-trees", "all"],
    tune_method: Literal["hyperopt", "manual"],
    early_stopping: Literal["auc", "aucpr", "none"],
    tune_loss: Literal[
        "roc_auc",
        "average_precision",
        "f1_weighted",
        "neg_log_loss",
        "partial_auc",
    ],
    n_features: int,
    n_samples: Optional[int],
    n_trials: int,
    n_jobs: int,
    seed: Optional[int],
) -> None:

    # Hyperopt space
    space = {
        "select__n_estimators": scope.int(
            hp.quniform("select__n_estimators", 50, 300, 10)
        ),
        "select__class_weight": hp.choice("select__class_weight", ["balanced", None]),
        "select__max_samples": hp.uniform("select__max_samples", 0.5, 1),
        "select__max_depth": scope.int(hp.quniform("select__max_depth", 2, 5, 1)),
        "select__min_samples_leaf": scope.int(
            hp.quniform("select__min_samples_leaf", 1, 20, 1)
        ),
        "select__ccp_alpha": hp.uniform("select__ccp_alpha", 0, 5),
        "cls__learning_rate": hp.uniform("cls__learning_rate", 0.01, 0.5),
        "cls__gamma": hp.uniform("cls__gamma", 0, 20),
        "cls__subsample": hp.uniform("cls__subsample", 0.6, 0.99),
        "cls__colsample_bytree": hp.uniform("cls__colsample_bytree", 0.5, 0.8),
        "cls__max_depth": scope.int(hp.quniform("cls__max_depth", 2, 31, 1)),
        "cls__reg_alpha": hp.quniform("cls__reg_alpha", 5, 20, 1),
        "cls__reg_lambda": hp.quniform("cls__reg_lambda", 0, 0.8, 1),
        "cls__min_child_weight": hp.quniform("cls__min_child_weight", 1, 15, 1),
        "cls__scale_pos_weight": hp.quniform("cls__scale_pos_weight", 1, 10, 1),
        "cls__n_estimators": scope.int(hp.quniform("cls__n_estimators", 10, 300, 10)),
        "cls__max_bin": scope.int(hp.quniform("cls__max_bin", 2, 200, 5)),
    }

    # partial AUC
    MAX_FPR = 0.1

    seed = int(seed) if seed is not None else random.randint(0, 10_000)

    # ------------------------- Load & preprocess -------------------------
    data = pd.read_parquet(train_path)

    # Shuffle (StratifiedGroupKFold retains global order otherwise)
    data = data.sample(frac=1.0, replace=False, random_state=seed).reset_index(
        drop=True
    )

    if n_samples:
        data = data.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    weight_col = getattr(DEFAULTS, "WEIGHT_COL", None)
    if weight_col and weight_col in data.columns:
        data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
        sample_weight_full = data[weight_col].astype(float)
    else:
        sample_weight_full = pd.Series(
            np.ones(len(data), dtype=float),
            index=data.index,
        )

    # Features/target
    X_full = data.drop(columns=DEFAULTS.COLUMNS_TO_DROP, errors="ignore")
    y_full = data[DEFAULTS.RESP_COL].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_idx: Optional[np.ndarray] = None
    val_idx: Optional[np.ndarray] = None
    train_sample_weight = sample_weight_full
    val_sample_weight: Optional[pd.Series] = None

    if early_stopping != "none":
        train_idx, val_idx = next(iter(skf.split(X_full, y_full)))
        X_train, y_train, X_val, y_val = (
            X_full.iloc[train_idx],
            y_full.iloc[train_idx],
            X_full.iloc[val_idx],
            y_full.iloc[val_idx],
        )
        train_sample_weight = sample_weight_full.iloc[train_idx]
        val_sample_weight = sample_weight_full.iloc[val_idx]
    else:
        X_train, y_train = X_full, y_full
        X_val, y_val = None, None
        val_sample_weight = None

    # Objective for Hyperopt
    def objective(params) -> Dict[str, Any]:
        feature_selector = get_feature_selector(
            reduce_method,
            n_features,
            n_jobs,
            seed,
            **get_select_params(params),
        )

        if reduce_method == "extra-trees":
            selector_fit_kwargs = {
                "sample_weight": train_sample_weight.loc[X_train.index]
            }
            features = feature_selector.fit(
                X_train,
                y_train,
                **selector_fit_kwargs,
            ).get_feature_names_out()
        else:
            features = feature_selector.fit(X_train, y_train).get_feature_names_out()

        X_train_transform = X_train[features]

        init_args: Dict[str, Any] = dict(
            objective="binary:logistic",
            tree_method="hist",
            base_score=y_train.mean(),
            random_state=seed,
            enable_categorical=True,
            verbosity=0,
            **get_classifier_params(params),
        )

        clf = xgb.XGBClassifier(**init_args)
        scoring = {
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "neg_log_loss": "neg_log_loss",
            "f1_weighted": "f1_weighted",
            "partial_auc": make_scorer(
                roc_auc_score,
                greater_is_better=True,
                max_fpr=MAX_FPR,
            ),
        }

        fit_params_cv: Dict[str, Any] = {}
        if train_sample_weight is not None:
            weight_key = "sample_weight"
            fit_params_cv[weight_key] = train_sample_weight.to_numpy(dtype=float)

        results = cross_validate(
            clf,
            X_train_transform,
            y_train,
            groups=None,
            scoring=scoring,
            cv=skf,
            n_jobs=n_jobs,
            pre_dispatch=n_jobs,
            return_estimator=True,
            error_score="raise",
            params=fit_params_cv,
        )

        summary = {
            "cv_roc_auc": np.array(results["test_roc_auc"]).mean(),
            "cv_average_precision": np.array(results["test_average_precision"]).mean(),
            "cv_neg_log_loss": np.array(results["test_neg_log_loss"]).mean(),
            "cv_f1_weighted": np.array(results["test_f1_weighted"]).mean(),
            "cv_partial_auc": np.array(results["test_partial_auc"]).mean(),
            "avg_fit_time": np.array(results["fit_time"]).mean(),
            "avg_score_time": np.array(results["score_time"]).mean(),
            "status": STATUS_OK,
        }

        loss_metric = tune_loss
        if tune_loss in ["roc_auc", "average_precision", "f1_weighted", "partial_auc"]:
            # maximize => minimize negative
            summary["loss"] = -summary[f"cv_{tune_loss}"]
        else:
            summary["loss"] = summary[f"cv_{tune_loss}"]

        # Introspect params of the fitted estimator
        params_ = results["estimator"][0].get_params()
        logger.info("CV params: %s\nSummary(loss): %s", params_, summary["loss"])

        return summary

    #  Choose parameters per tune-method
    trials: Optional[Trials] = None
    params: Optional[Dict[str, Any]] = None

    if tune_method == "hyperopt":
        trials = Trials()
        params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            early_stop_fn=no_progress_loss(percent_increase=0.5),
            rstate=np.random.default_rng(seed),
            show_progressbar=True,
            # verbose=True,
        )
        params = space_eval(space, params)
        print("Selected Parameters: ", params)
    elif tune_method == "manual":
        params = {f"cls_{k}": v for k, v in DEFAULTS.MANUAL_MODEL_PARAMS.items()}

    if not params:
        raise RuntimeError("Hyperparameter tuning failed.")

    # Rebuild selector with chosen params
    feature_selector = get_feature_selector(
        reduce_method, n_features, n_jobs, seed, **get_select_params(params)
    )

    init_args = dict(
        objective="binary:logistic",
        tree_method="hist",
        base_score=y_full.mean(),
        random_state=seed,
        eval_metric="auc",
        enable_categorical=True,
        verbosity=0,
        **get_classifier_params(params),
    )

    # ------------------------------ MLflow run ------------------------------
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Build model (grouped or single)
        model = xgb.XGBClassifier(**init_args)

        print("Model parameters:", model.get_params())

        fit_params: Dict[str, Any] = {}
        weight_key = "sample_weight"

        # Feature selection on the full data for final fit
        selector_fit_kwargs_final: Dict[str, Any] = {}
        if reduce_method in {"extra-trees"}:
            selector_fit_kwargs_final["sample_weight"] = sample_weight_full

        feature_selector.fit(X_full, y_full, **selector_fit_kwargs_final)

        features = feature_selector.get_feature_names_out()
        X_select = X_full[features]
        print("Selected features: %s", features)

        # Persist selected features for reproducibility.
        mlflow.log_dict(
            {"selected_features": features},
            artifact_file="selected_features.json",
        )

        if early_stopping != "none" and train_idx is not None and val_idx is not None:
            X_train_select = X_select.iloc[train_idx]
            y_train_select = y_full.iloc[train_idx]
            X_val_select = X_select.iloc[val_idx]
            y_val_select = y_full.iloc[val_idx]
            train_weight_array = train_sample_weight.to_numpy(dtype=float)
            fit_params[weight_key] = train_weight_array

            fit_params.update(
                {
                    "eval_set": [(X_val_select, y_val_select)],
                    **(
                        {
                            "sample_weight_eval_set": [
                                val_sample_weight.to_numpy(dtype=float)
                            ]
                        }
                        if val_sample_weight is not None
                        else {}
                    ),
                    **(
                        {"sample_weight": train_sample_weight.to_numpy(dtype=float)}
                        if train_sample_weight is not None
                        else {}
                    ),
                }
            )
            fit_params.setdefault("verbose", False)

            model.fit(X_train_select, y_train_select, **fit_params)
        else:
            fit_params[weight_key] = sample_weight_full.to_numpy(dtype=float)
            model.fit(X_select, y_full, **fit_params)

        signature = infer_signature(X_select, y_full)

        # Drop params which are themselves estimators (e.g., steps in pipeline)
        from sklearn.base import BaseEstimator as _BaseEstimator

        mlflow_params = {
            k: v
            for k, v in model.get_params().items()
            if not isinstance(v, _BaseEstimator) and k != "steps"
        }

        # Attach extra metrics from tuning or objective
        if trials:
            metrics = {
                k: v
                for k, v in trials.best_trial.get("result", {}).items()
                if isinstance(v, (float, int))
            }
        else:
            metrics = {}

        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(metrics)

        model_name = "_".join(
            [
                (
                    "xgb_all"
                    if reduce_method == "all"
                    else f"xgb_select_{n_features}_by_{reduce_method}"
                ),
                f"sample_{n_samples}" if n_samples else "",
                f"stop_{early_stopping}" if early_stopping != "none" else "",
                f"tune_{tune_method}",
            ]
        ).strip("_")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_artifacts",
            signature=signature,
            input_example=X_select.sample(min(1000, len(X_select))),
            registered_model_name=model_name,
            serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "reduce_method": reduce_method,
                "tune_method": tune_method,
                "tune_loss": tune_loss,
                "early_stopping": early_stopping,
                "n_features": n_features,
                "n_samples": n_samples or "all",
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "seed": seed,
            },
        )
