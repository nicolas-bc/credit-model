import numpy as np
import pandas as pd
from typing import List
from sklearn.base import TransformerMixin


class LogTransformer(TransformerMixin):
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.features:
            X[feature] = np.log1p(X[feature])
        return X


class FeatureRemoval(TransformerMixin):
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.features, axis=1)


def create__general_lost_sum__over__general_limite_de_cred_sum(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["general_lost_sum__over__general_limite_de_cred_sum"] = (
        df["general_lost_sum"] / df["general_limite_de_cred_sum"]
    )
    return df


def create__general_vencido_sum__over__general_limite_de_cred_sum(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["general_vencido_sum__over__general_limite_de_cred_sum"] = (
        df["general_vencido_sum"] / df["general_limite_de_cred_sum"]
    )
    return df


class FeatureCreation(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = create__general_lost_sum__over__general_limite_de_cred_sum(X)
        X = create__general_vencido_sum__over__general_limite_de_cred_sum(X)
        return X
