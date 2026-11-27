"""GripModel: Random Forest classifier for grip_lost."""
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from core.cleaning import scale_features

logger = logging.getLogger(__name__)

TARGET = "grip_lost"
DROP = ["Num", "Timestamp", "cycle", "Robot_ProtectiveStop", "temp_band", TARGET]


class GripModel:
    def __init__(self, n_estimators=100, test_size=0.2, scale=False, random_state=42):
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.scale = scale
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", random_state=random_state)
        self.features = []

    def split(self, df: pd.DataFrame):
        """Group split by cycle so one work cycle is never in both train and test."""
        X = df.drop(columns=[c for c in DROP if c in df.columns])
        y = df[TARGET].astype(int)
        self.features = list(X.columns)
        if "cycle" in df.columns and df["cycle"].nunique() > 1:
            gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size,
                                    random_state=self.random_state)
            tr, te = next(gss.split(X, y, groups=df["cycle"]))
            return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        return train_test_split(X, y, test_size=self.test_size,
                                stratify=y, random_state=self.random_state)

    def fit_evaluate(self, df: pd.DataFrame) -> dict:
        X_tr, X_te, y_tr, y_te = self.split(df)
        if self.scale:
            X_tr, X_te, _ = scale_features(X_tr, X_te)
        self.model.fit(X_tr, y_tr)
        pred = self.model.predict(X_te)
        results = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "baseline_accuracy": 1 - y_te.mean(),     # "always False"
            "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]),
        }
        logger.info("Model results: %s", {k: v for k, v in results.items() if k != "confusion_matrix"})
        return results

    def feature_importances(self, top=10) -> pd.Series:
        s = pd.Series(self.model.feature_importances_, index=self.features)
        return s.sort_values(ascending=False).head(top)
