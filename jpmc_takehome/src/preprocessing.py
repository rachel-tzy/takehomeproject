from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ALL_FEATURES, DEFAULTS, NUMERICAL_FEATURES

# Default locations for assets.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "assets/raw"
SAVE_DIR = BASE_DIR / "assets/processed"

# Columns to fill with the "Missing" placeholder, based on the EDA notebook.
MISSING_CATEGORICAL_IMPUTES: Dict[str, str] = {
    "migration prev res in sunbelt": "Missing",
    "migration code-change in msa": "Missing",
    "migration code-change in reg": "Missing",
    "migration code-move within reg": "Missing",
    "country of birth father": "Missing",
    "country of birth mother": "Missing",
    "country of birth self": "Missing",
    "hispanic origin": "Missing",
    "state of previous residence": "Missing",
}


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def load_raw_data(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Read the raw census data using the column metadata file."""
    columns_path = data_dir / "census-bureau.columns"
    data_path = data_dir / "census-bureau.data"

    if not columns_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            "Could not locate the census data files. "
            f"Expected to find {columns_path} and {data_path}."
        )

    columns = columns_path.read_text(encoding="utf-8").strip().splitlines()

    df = pd.read_csv(
        data_path,
        names=columns,
        na_values=["?", " ?", "None", ""],
        skipinitialspace=True,
        engine="c",
        encoding="latin-1",
    )

    # Align label values with the replacements used in the EDA.
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .replace({"- 50000": 0, "50000+": 1})
        .astype(int)
    )

    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived numeric features shared across tasks."""
    engineered = df.copy()

    if {"capital gains", "wage per hour"}.issubset(engineered.columns):
        wage = pd.to_numeric(engineered["wage per hour"], errors="coerce")
        ratios = wage.replace({0: np.nan})
        engineered["ratio_capital_gain_wage"] = (
            (pd.to_numeric(engineered["capital gains"], errors="coerce") / ratios)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    if {"capital gains", "dividends from stocks"}.issubset(engineered.columns):
        gains = pd.to_numeric(engineered["capital gains"], errors="coerce")
        dividends = pd.to_numeric(engineered["dividends from stocks"], errors="coerce")
        totals = (gains + dividends).replace({0: np.nan})
        engineered["ratio_dividend_income"] = (
            (dividends / totals).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    if {"weeks worked in year", "age"}.issubset(engineered.columns):
        weeks = pd.to_numeric(engineered["weeks worked in year"], errors="coerce")
        age = pd.to_numeric(engineered["age"], errors="coerce").replace({0: np.nan})
        engineered["employment_intensity"] = (
            (weeks / age).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    return engineered


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean label, impute, and enforce dtypes in line with the EDA pipeline."""
    processed = _add_ratio_features(df)

    # Fill high-missing categorical columns with a shared placeholder.
    processed = processed.fillna(MISSING_CATEGORICAL_IMPUTES)

    # Split feature groups into numeric and categorical for dtype handling.
    categorical_columns = [col for col in ALL_FEATURES if col not in NUMERICAL_FEATURES]

    # Ensure numeric columns are numeric, coercing unexpected strings to NaN.
    for column in (col for col in NUMERICAL_FEATURES if col in processed.columns):
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    # Fill remaining categorical gaps and cast to 'category' dtype for pandas.
    categorical_present = [
        col for col in categorical_columns if col in processed.columns
    ]
    for column in categorical_present:
        processed[column] = processed[column].astype("category")
    processed_unmapped = processed.copy()

    for column in categorical_present:
        categories = sorted(processed[column].cat.categories.tolist())
        mapping = DEFAULTS.register_categories(column, categories)
        processed[column] = processed[column].map(mapping).astype("int16")

    # Include optional auxiliary columns
    extra_columns: list[str] = []
    weight_col = getattr(DEFAULTS, "WEIGHT_COL", None)
    if weight_col and weight_col in processed.columns:
        processed[weight_col] = pd.to_numeric(
            processed[weight_col], errors="coerce"
        ).fillna(1.0)
        extra_columns.append(weight_col)

    # Keep recognised feature columns, auxiliary columns, and the response.
    retained_columns = (
        [col for col in ALL_FEATURES if col in processed.columns]
        + extra_columns
        + [DEFAULTS.RESP_COL]
    )
    processed = processed[retained_columns].copy()

    return processed, processed_unmapped


def split_train_test(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the processed data into train and test partitions."""
    label = DEFAULTS.RESP_COL
    if label not in df.columns:
        raise KeyError("Label column missing from dataframe prior to splitting.")

    X = df.drop(columns=label)
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = X_train.assign(**{label: y_train.values})
    test_df = X_test.assign(**{label: y_test.values})
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed_datasets(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist processed splits to parquet for downstream training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "data.parquet"
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    pd.concat([train_df, test_df]).to_parquet(all_path, index=False)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)


# @click.group()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="Directory containing the raw census data files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=SAVE_DIR,
    show_default=True,
    help="Directory where processed parquet files will be written.",
)
@click.option(
    "--test-size",
    type=float,
    default=DEFAULT_TEST_SIZE,
    show_default=True,
    help="Fraction of the data reserved for the test split.",
)
@click.option(
    "--random-state",
    type=int,
    default=DEFAULT_RANDOM_STATE,
    show_default=True,
    help="Random seed used for the stratified train/test split.",
)
@click.command()
def main(
    data_dir: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> None:
    """Generate train/test datasets."""
    raw_df = load_raw_data(data_dir=data_dir)
    processed_df, processed_unmapped = preprocess_dataframe(raw_df)
    processed_unmapped.to_parquet(output_dir / "data_unmapped.parquet", index=False)
    train_df, test_df = split_train_test(
        processed_df, test_size=test_size, random_state=random_state
    )
    save_processed_datasets(output_dir, train_df, test_df)


if __name__ == "__main__":
    main()
