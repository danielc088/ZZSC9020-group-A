"""Real-data contract check without fitting models or inspecting final scores.

Run from repository root: python -B tests/validate_task11.py
Writes only results/task11_validation.json; original data are read-only.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.forecast_data import (
    FeatureConfig, PROTOCOL_VERSION, build_fold, default_splits,
    load_prepared_data, load_school_terms,
)


def main():
    config = FeatureConfig()
    data = load_prepared_data(config=config)
    terms = load_school_terms()
    rows = []
    for _, split in default_splits().iterrows():
        main_fold = build_fold(data, split, terms, "analogue", config)
        diagnostic = build_fold(data, split, terms, "observed_diagnostic", config)
        climate = build_fold(data, split, terms, "climatology", config)
        pd.testing.assert_frame_equal(main_fold.X_train, diagnostic.X_train)
        pd.testing.assert_frame_equal(main_fold.X_train, climate.X_train)
        assert main_fold.X_future.index.equals(diagnostic.X_future.index)
        assert main_fold.X_future.index.equals(climate.X_future.index)
        assert (main_fold.future_metadata.temperature_source_time.dropna()
                < pd.Timestamp(split["origin"])).all()
        for matrix in (main_fold.X_train, main_fold.X_future, diagnostic.X_future, climate.X_future):
            assert np.isfinite(matrix.to_numpy()).all()
        rows.append({key: main_fold.metadata[key] for key in (
            "fold_id", "train_rows", "forecast_rows", "analogue_year",
        )})
        print(split["fold_id"], "passed", flush=True)
    result = {key: main_fold.metadata[key] for key in (
        "python", "dependencies", "config", "source_sha256",
        "school_calendar_sha256", "implementation_sha256", "timezone_status",
    )}
    result.update({"protocol": PROTOCOL_VERSION,
                   "feature_columns": list(main_fold.X_future),
                   "weather_modes_verified": ["analogue", "observed_diagnostic", "climatology"],
                   "real_data_folds_verified_in_all_modes": rows,
                   "models_trained": 0})
    (ROOT / "results" / "task11_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("All ten folds passed in all three weather modes; no models fitted.")


if __name__ == "__main__":
    main()
