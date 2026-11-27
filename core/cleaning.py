"""BaseProcessor and Cleaner (inheritance)."""
import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ID_COLUMNS = ["Num", "Timestamp", "cycle"]
TARGET = "grip_lost"


class BaseProcessor:
    """Common behaviour for every processing step: holds data and a log."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log = []

    def _record(self, message: str):
        self.log.append(message)
        logger.info(message)

    def get_log(self):
        return list(self.log)


class Cleaner(BaseProcessor):
    """Cleans data: names, empty rows, missing values, duplicates, new feature."""

    def fix_column_names(self):
        if "Temperature_T0" in self.df.columns:
            self.df = self.df.rename(columns={"Temperature_T0": "Temperature_J0"})
            self._record("Renamed Temperature_T0 -> Temperature_J0")
        return self

    def drop_empty_rows(self):
        sensors = self.sensor_columns()
        before = len(self.df)
        self.df = self.df.dropna(subset=sensors, how="all")
        self._record(f"Dropped {before - len(self.df)} rows where every sensor was empty")
        return self

    def fill_missing(self):
        sensors = self.sensor_columns() + ["Robot_ProtectiveStop"]
        for col in [c for c in sensors if c in self.df.columns]:
            n = int(self.df[col].isna().sum())
            if n:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self._record(f"Filled {n} missing values in {col} with median")
        return self

    def drop_duplicates(self):
        cols = [c for c in self.df.columns if c not in ("Num", "Timestamp")]
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=cols)
        self._record(f"Removed {before - len(self.df)} duplicate rows")
        return self

    def add_temperature_band(self):
        """Categorical feature: Low / Medium / High from mean joint temperature."""
        temp_cols = [c for c in self.df.columns if c.startswith("Temperature_")]
        mean_temp = self.df[temp_cols].mean(axis=1)
        self.df["temp_band"] = pd.qcut(mean_temp, 3, labels=["Low", "Medium", "High"]).astype(str)
        self._record("Created categorical column temp_band (Low/Medium/High)")
        return self

    def encode_band(self):
        mapping = {"Low": 0, "Medium": 1, "High": 2}
        self.df["temp_band_code"] = self.df["temp_band"].map(mapping)
        self._record("Encoded temp_band as temp_band_code (0/1/2)")
        return self

    def sensor_columns(self):
        skip = set(ID_COLUMNS + [TARGET, "Robot_ProtectiveStop", "temp_band", "temp_band_code"])
        return [c for c in self.df.columns
                if c not in skip and pd.api.types.is_numeric_dtype(self.df[c])]

    def clean(self) -> pd.DataFrame:
        """Run all cleaning steps in order and return the cleaned table."""
        try:
            (self.fix_column_names().drop_empty_rows().fill_missing()
                 .drop_duplicates().add_temperature_band().encode_band())
        except Exception as exc:
            logger.error("Cleaning failed: %s", exc)
            raise
        self.df = self.df.reset_index(drop=True)
        self._record(f"Final table: {self.df.shape[0]} rows, {int(self.df.isna().sum().sum())} missing values")
        return self.df


def scale_features(train: pd.DataFrame, test: pd.DataFrame):
    """Fit scaler on TRAIN only (avoids data leakage) and apply to both."""
    scaler = StandardScaler().fit(train)
    return (pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index),
            pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index),
            scaler)
