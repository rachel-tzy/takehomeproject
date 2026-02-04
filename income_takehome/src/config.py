from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

# Core XGBoost hyperparameters that work well with the census income dataset.
DEFAULT_XGB_PARAMS: Dict[str, float] = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "enable_categorical": True,
}

MANUAL_XGB_PARAMS: Dict[str, float] = {
    **DEFAULT_XGB_PARAMS,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "max_depth": 8,
    "n_estimators": 400,
}

ALL_FEATURES = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
    "ratio_capital_gain_wage",
    "ratio_dividend_income",
    "employment_intensity",
    "class of worker",
    "major industry code",
    "detailed industry recode",
    "major occupation code",
    "detailed occupation recode",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "education",
    "enroll in edu inst last wk",
    "marital stat",
    "detailed household and family stat",
    "detailed household summary in household",
    "family members under 18",
    "race",
    "hispanic origin",
    "sex",
    "citizenship",
    "region of previous residence",
    "state of previous residence",
    "live in this house 1 year ago",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "migration prev res in sunbelt",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "own business or self employed",
    "tax filer stat",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
]

NUMERICAL_FEATURES = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
    "ratio_capital_gain_wage",
    "ratio_dividend_income",
    "employment_intensity",
]


def _stable_category_mapping(categories: Iterable[str]) -> Dict[str, int]:
    """Return a deterministic mapping of category label to integer code."""
    unique = dict.fromkeys(categories)  # preserves declared order
    ordered = sorted(unique, key=lambda value: (value != "Missing", value))
    return {value: idx for idx, value in enumerate(ordered)}


@dataclass
class Config:
    """Central place for column metadata and baseline model parameters."""

    RESP_COL: str = "label"
    WEIGHT_COL: Optional[str] = "weight"
    COLUMNS_TO_DROP: List[str] = field(
        default_factory=lambda: ["label", "weight", "year"]
    )
    DEFAULT_MODEL_PARAMS: Dict[str, float] = field(
        default_factory=lambda: {**DEFAULT_XGB_PARAMS}
    )
    MANUAL_MODEL_PARAMS: Dict[str, float] = field(
        default_factory=lambda: {**MANUAL_XGB_PARAMS}
    )
    CATEGORY_MAPPINGS: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def register_categories(
        self, column: str, categories: Iterable[str]
    ) -> Dict[str, int]:
        """Persist a reproducible mapping for a categorical column."""
        mapping = _stable_category_mapping(categories)
        existing = self.CATEGORY_MAPPINGS.get(column)
        if existing and existing != mapping:
            raise ValueError(
                f"Inconsistent category mapping detected for '{column}'. "
                "Ensure preprocessing yields deterministic category sets."
            )
        self.CATEGORY_MAPPINGS[column] = mapping
        return mapping


DEFAULTS = Config()
