import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cleaning import Cleaner
from core.loader import DataLoader
from core.model import GripModel

DATA = Path(__file__).resolve().parents[1] / "data" / "dataset_02052023.xlsx"


def test_loader_rejects_wrong_type(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("hello")
    with pytest.raises(ValueError):
        DataLoader(f).load()


def test_loader_rejects_empty_csv(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("")
    with pytest.raises(ValueError):
        DataLoader(f).load()


def test_cleaner_no_missing_no_duplicates():
    df = DataLoader(DATA).load()
    clean = Cleaner(df).clean()
    assert clean.isna().sum().sum() == 0
    assert not clean.drop(columns=["Num", "Timestamp"]).duplicated().any()


def test_model_metrics_between_0_and_1():
    df = Cleaner(DataLoader(DATA).load()).clean().sample(1500, random_state=1)
    res = GripModel(n_estimators=20).fit_evaluate(df)
    for k in ("accuracy", "precision", "recall", "f1"):
        assert 0 <= res[k] <= 1
