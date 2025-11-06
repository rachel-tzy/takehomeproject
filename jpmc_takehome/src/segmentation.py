from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import click
import numpy as np
import pandas as pd

from .config import ALL_FEATURES, NUMERICAL_FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "assets/processed/data_unmapped.parquet"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results/segmentation_rules"


def weighted_quantile(values: np.ndarray, q: float, w: np.ndarray) -> float:
    """Weighted quantile (q in [0,1])."""
    values = np.asarray(values, dtype=float)
    w = np.asarray(w, dtype=float)
    m = ~np.isnan(values)
    values, w = values[m], w[m]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values, w = values[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return float(np.interp(q, cdf, values))


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    if np.sum(w) == 0:
        return np.nan
    return float(np.average(x, weights=w))


def num(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype="float64")
    return pd.to_numeric(d[col], errors="coerce")


def cat(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d.columns:
        return pd.Series("", index=d.index, dtype="string")
    return d[col].astype("string")


@dataclass(frozen=True)
class Rule:
    name: str
    priority: int
    predicate: Callable[[pd.DataFrame], pd.Series]  # returns boolean mask


def apply_rules(df: pd.DataFrame, default_segment: str = "Other") -> pd.Series:
    """Apply rules by descending priority, first match wins; fill unmatched with default."""

    # initialize segmentation column if not exists
    if "segmentation" not in df.columns:
        df["segmentation"] = np.nan

    # Threshold constants (edit as needed)
    TH_CHILDREN = 16
    TH_HIGH_WAGE = 30.0
    TH_MID_WAGE = 20.0
    TH_LOW_WAGE = 12.0
    TH_FULL_WEEKS = 40
    TH_LONG_WEEKS = 48
    TH_SHORT_WEEKS = 20
    TH_HIGH_EMP_INT = 0.9
    TH_LOW_EMP_INT = 0.3

    # Children / Adults
    df["segmentation"] = df["segmentation"].mask(
        df["segmentation"].isna() & (df["age"] < TH_CHILDREN),
        "Children",
    )

    # Capital / Dividend income present
    df["segmentation"] = df["segmentation"].mask(
        df["segmentation"].isna()
        & ((df["capital gains"] > 0) | (df["dividends from stocks"] > 0)),
        "CapitalIncomePresent",
    )

    # High-wage full-year worker
    df["segmentation"] = df["segmentation"].mask(
        df["segmentation"].isna()
        & (
            (df["wage per hour"] >= TH_HIGH_WAGE)
            & (df["weeks worked in year"] >= TH_FULL_WEEKS)
        ),
        "HighWage_FullTime",
    )

    # Low-wage but full year worker
    df["segmentation"] = df["segmentation"].mask(
        df["segmentation"].isna()
        & (
            (df["wage per hour"] < TH_LOW_WAGE)
            & (df["weeks worked in year"] >= TH_FULL_WEEKS)
        ),
        "LowWage_FullTime",
    )

    # Part-time or short-year worker
    df["segmentation"] = df["segmentation"].mask(
        df["segmentation"].isna()
        & (
            (
                df["full or part time employment stat"].isin(
                    [
                        "PT for econ reasons usually FT",
                        "PT for econ reasons usually PT",
                        "PT for non-econ reasons usually FT",
                        "Unemployed part- time",
                    ]
                )
            )
            | (df["weeks worked in year"] <= TH_SHORT_WEEKS)
            | (df["employment_intensity"] <= TH_LOW_EMP_INT)
        ),
        "PartTime_or_ShortWeeks",
    )

    # High employment intensity
    df["segmentation"] = df["segmentation"].mask(
        df["segmentation"].isna() & (df["employment_intensity"] >= TH_HIGH_EMP_INT),
        "HighEmploymentIntensity",
    )

    # fallback (P2 = equal-level, so fill anything unmatched)
    df["segmentation"] = df["segmentation"].fillna(default_segment)
    return df


def weighted_profiles(
    df: pd.DataFrame,
    weight: pd.Series | np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    w = np.asarray(weight, dtype=float)
    total_w = w.sum() if w.sum() > 0 else len(w)
    out = []
    segments = df["segmentation"]
    for s in segments.unique():
        m = (segments == s).to_numpy()
        w_seg = w[m]
        row = dict(
            segment=s,
            pop_weight=float(w_seg.sum()),
            pop_share=float(w_seg.sum() / total_w if total_w > 0 else m.mean()),
        )
        # numeric means
        for col in numeric_cols:
            row[f"mean_{col}"] = wmean(
                pd.to_numeric(df.loc[m, col], errors="coerce").to_numpy(), w_seg
            )
        # top categorical levels (weighted proportion)
        for col in categorical_cols:
            cats = df.loc[m, col].astype("string")
            wf = (
                pd.DataFrame({"cat": cats, "w": w_seg})
                .groupby("cat", sort=False)["w"]
                .sum()
            )
            wf = wf.sort_values(ascending=False)
            top3 = (
                "; ".join(
                    [f"{k} ({float(v/w_seg.sum()):.1%})" for k, v in wf.head(3).items()]
                )
                if w_seg.sum() > 0
                else ""
            )
            row[f"top_{col}"] = top3
        out.append(row)
    return pd.DataFrame(out).sort_values("pop_weight", ascending=False)


@click.command()
@click.option(
    "--data-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="Parquet dataset containing processed census features.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, dir_okay=True),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory where segmentation artifacts will be written.",
)
@click.option(
    "--weight-col",
    default="weight",
    show_default=True,
    help="Name of the sample-weight column.",
)
def main(data_path: Path, output_dir: Path, weight_col: str) -> None:
    """Run the rule-based segmentation pipeline against a processed dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data parquet not found at {data_path}")

    df = pd.read_parquet(data_path)
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' missing from dataset.")

    weight = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy()

    df = apply_rules(df, default_segment="Other")

    numeric_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    if weight_col in numeric_cols:
        numeric_cols.remove(weight_col)
    categorical_cols = [
        col for col in ALL_FEATURES if col in df.columns and col not in numeric_cols
    ]

    profiles = weighted_profiles(df, weight, numeric_cols, categorical_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_path = output_dir / "rule_segments.parquet"
    profiles_path = output_dir / "segment_profiles.parquet"

    df.to_parquet(segments_path, index=False)
    profiles.to_parquet(profiles_path, index=False)

    summary = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "segments_file": str(segments_path),
        "profiles_file": str(profiles_path),
    }
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
