"""Reusable pipeline: load -> clean -> analyze -> model -> save."""
import logging
from pathlib import Path

from core.analysis import Analyzer
from core.cleaning import Cleaner
from core.loader import DataLoader
from core.model import GripModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean(path):
    df = DataLoader(path).load()
    cleaner = Cleaner(df)
    clean_df = cleaner.clean()
    return clean_df, cleaner.get_log()


def analyze(clean_df):
    a = Analyzer(clean_df)
    return {"describe": a.describe(), "correlation": a.correlation(),
            "outliers": a.outlier_counts()}


def train_model(clean_df, n_estimators=100, test_size=0.2, scale=False):
    gm = GripModel(n_estimators=n_estimators, test_size=test_size, scale=scale)
    results = gm.fit_evaluate(clean_df)
    return results, gm.feature_importances()


def save_clean(clean_df, out_path="reports/cleaned_data.csv"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    logger.info("Saved cleaned data to %s", out_path)
    return out_path


def run_all(path="data/dataset_02052023.xlsx"):
    clean_df, log = load_and_clean(path)
    stats = analyze(clean_df)
    results, importances = train_model(clean_df)
    save_clean(clean_df)
    return {"clean": clean_df, "log": log, "stats": stats,
            "results": results, "importances": importances}


if __name__ == "__main__":
    out = run_all()
    for k, v in out["results"].items():
        print(k, v)
