"""Analyzer: descriptive statistics, correlations and IQR outlier detection."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric = df.select_dtypes("number")

    def describe(self) -> pd.DataFrame:
        return self.numeric.describe().T

    def correlation(self) -> pd.DataFrame:
        return self.numeric.corr()

    def outlier_counts(self, columns=None) -> pd.Series:
        """Count values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for each column."""
        cols = columns or list(self.numeric.columns)
        counts = {}
        for col in cols:
            low, high = self.iqr_bounds(col)
            counts[col] = int(((self.df[col] < low) | (self.df[col] > high)).sum())
        return pd.Series(counts, name="outliers")

    def iqr_bounds(self, col):
        q1, q3 = self.df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    def remove_outliers(self, columns) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        for col in columns:
            low, high = self.iqr_bounds(col)
            mask &= self.df[col].between(low, high)
        logger.info("Outlier removal kept %s of %s rows", int(mask.sum()), len(self.df))
        return self.df[mask]

    def grip_loss_by(self, column) -> pd.Series:
        return self.df.groupby(column)["grip_lost"].mean()
