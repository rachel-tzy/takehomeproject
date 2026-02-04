from __future__ import annotations

import numbers
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
)
from typing_extensions import Self
from xgboost import XGBClassifier

from .config import ALL_FEATURES


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None) -> Self:
        return self

    def transform(self, X, y=None) -> Any:
        return X

    def get_feature_names_out(self, features=ALL_FEATURES):
        return features


def get_feature_selector(
    reduce_method: str,
    n_features: int,
    n_jobs: int,
    seed: int,
    **kwargs,
):
    """
    Factory to obtain a feature selector by name.
    """
    if reduce_method == "minfo":
        k_nn_features = kwargs.get("k_nn_features", n_features)
        return SelectKBest(mutual_info_classif, k=k_nn_features)

    elif reduce_method == "extra-trees":
        return SelectFromModel(
            ExtraTreesClassifier(
                bootstrap=True,
                n_jobs=n_jobs,
                random_state=seed,
                **kwargs,
            ),
            threshold=-np.inf,
            max_features=n_features,
        )

    elif reduce_method == "all":
        return IdentityTransformer()

    else:
        return IdentityTransformer()
