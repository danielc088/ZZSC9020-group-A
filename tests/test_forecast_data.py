"""Run from repository root: python -B -m unittest discover -s tests -v."""

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.forecast_data import (
    FeatureConfig, build_fold, calendar_features, default_splits,
    evaluation_frame, export_fold, load_prepared_data, load_school_terms,
    predictions_frame, _climatology,
)


class ForecastContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terms = load_school_terms()
        index = pd.date_range("2018-01-01", "2021-01-01", freq="30min", inclusive="left", name="DATETIME")
        cls.data = pd.DataFrame({
            "TOTALDEMAND": 7000 + 1000 * np.sin(np.arange(len(index)) / 48),
            "temperature_observed_c": 18 + 10 * np.sin(np.arange(len(index)) / 480),
            "temperature_source_time": index,
            "temperature_offset_minutes": 0.0,
        }, index=index)
        cls.split = {"fold_id": "final_test_20200101", "role": "final_test",
                     "train_start": "2018-01-01", "origin": "2020-01-01",
                     "forecast_end_exclusive": "2021-01-01"}
        cls.fold = build_fold(cls.data, cls.split, cls.terms)

    def test_default_development_never_overlaps_final_test(self):
        splits = default_splits()
        dev = splits[splits.role == "development"]
        final = splits[splits.role == "final_test"]
        self.assertEqual(len(dev), 8)
        self.assertEqual(len(final), 2)
        self.assertLessEqual(dev.forecast_end_exclusive.max(), final.origin.min())
        self.assertEqual(set(dev.origin.dt.month), {1, 4, 7, 10})

    def test_checked_in_manifests_match_generators(self):
        from src.forecast_data import ROOT, public_holiday_calendar
        folder = ROOT / "data" / "NSW" / "calendar"
        stored = pd.read_csv(folder / "split_manifest.csv",
                             parse_dates=["train_start", "origin", "forecast_end_exclusive"])
        assert_frame_equal(stored, default_splits(), check_dtype=False)
        dates = pd.read_csv(folder / "public_holidays.csv")
        # The saved calendar uses Australian English ("Labour Day"). Keep this
        # comparison independent of the terminal's holiday-name language.
        with patch.dict(os.environ, {"LANGUAGE": "en_AU"}):
            expected = public_holiday_calendar(range(2010, 2022))
        self.assertEqual(dict(zip(pd.to_datetime(dates.date).dt.date, dates.holiday_name)), expected)

    def test_invalid_training_demand_and_partial_month_fail(self):
        changed = self.data.copy()
        changed.iloc[0, changed.columns.get_loc("TOTALDEMAND")] = np.nan
        with self.assertRaisesRegex(ValueError, "Training demand"):
            build_fold(changed, self.split, self.terms)
        with self.assertRaisesRegex(ValueError, "Forecast end"):
            build_fold(self.data, dict(self.split, forecast_end_exclusive="2020-02-02"), self.terms)

    def test_boundaries_full_leap_year_and_same_schema(self):
        fold = self.fold
        self.assertLess(fold.X_train.index.max(), pd.Timestamp("2020-01-01"))
        self.assertEqual(fold.X_future.index.min(), pd.Timestamp("2020-01-01"))
        self.assertEqual(fold.X_future.index.max(), pd.Timestamp("2020-12-31 23:30"))
        self.assertEqual(len(fold.X_future), 366 * 48)
        self.assertEqual(list(fold.X_train), list(fold.X_future))
        self.assertNotIn("TOTALDEMAND", fold.X_future)
        self.assertFalse(any(c.startswith("forecast_") for c in fold.X_future))
        self.assertEqual(set(fold.future_metadata.lead_month), set(range(1, 13)))

    def test_future_demand_and_weather_cannot_change_analogue_inputs(self):
        changed = self.data.copy()
        future = changed.index >= pd.Timestamp(self.split["origin"])
        changed.loc[future, "TOTALDEMAND"] = -999999
        changed.loc[future, "temperature_observed_c"] = 9999
        other = build_fold(changed, self.split, self.terms)
        assert_frame_equal(self.fold.X_train, other.X_train)
        assert_frame_equal(self.fold.X_future, other.X_future)

    def test_diagnostic_changes_future_weather_only(self):
        diagnostic = build_fold(self.data, self.split, self.terms, "observed_diagnostic")
        assert_frame_equal(self.fold.X_train, diagnostic.X_train)
        actual = self.data.loc[diagnostic.X_future.index, "temperature_observed_c"]
        np.testing.assert_allclose(diagnostic.future_metadata.temperature_input_c, actual)

    def test_analogue_sources_precede_origin_and_leap_mapping(self):
        meta = self.fold.future_metadata
        self.assertTrue((meta.temperature_source_time.dropna() < pd.Timestamp("2020-01-01")).all())
        self.assertEqual(int(meta.leap_day_mapped_to_feb28.sum()), 48)
        self.assertEqual(meta.loc["2020-02-29 12:00", "temperature_source_time"],
                         pd.Timestamp("2019-02-28 12:00"))

    def test_nearest_weather_crossing_origin_is_not_available_to_training(self):
        changed = self.data.copy()
        stamp = pd.Timestamp("2019-12-31 23:30")
        changed.loc[stamp, "temperature_source_time"] = pd.Timestamp("2020-01-01")
        changed.loc[stamp, "temperature_observed_c"] = 9999
        fold = build_fold(changed, self.split, self.terms)
        self.assertTrue(fold.train_quality.loc[stamp, "temperature_imputed"])
        self.assertLess(fold.train_quality.loc[stamp, "temperature_input_c"], 40)

    def test_missing_weather_uses_training_only(self):
        changed = self.data.copy()
        changed.loc["2019-07-01":"2019-07-04", "temperature_observed_c"] = np.nan
        first = build_fold(changed, self.split, self.terms)
        changed.loc["2020":, "temperature_observed_c"] = 10000
        second = build_fold(changed, self.split, self.terms)
        assert_frame_equal(first.X_train, second.X_train)
        assert_frame_equal(first.X_future, second.X_future)
        self.assertGreater(first.metadata["train_temperature_imputed_rows"], 0)
        self.assertTrue(np.isfinite(first.X_future).all().all())

    def test_no_usable_training_weather_fails(self):
        changed = self.data.copy()
        changed.loc[changed.index < pd.Timestamp("2020-01-01"), "temperature_observed_c"] = np.nan
        with self.assertRaisesRegex(ValueError, "No reliable"):
            build_fold(changed, self.split, self.terms)

    def test_configured_window_used_for_training_donor_and_climatology_without_future_leakage(self):
        changed = self.data.copy()
        stamp = pd.Timestamp("2019-02-15 10:00")
        changed.loc[stamp, "temperature_observed_c"] = np.nan
        for width in (3, 5):
            config = FeatureConfig(climatology_window_days=width)
            analogue = build_fold(changed, self.split, self.terms, config=config)
            climate = build_fold(changed, self.split, self.terms, "climatology", config)
            known = changed.loc[changed.index < pd.Timestamp("2020-01-01"), "temperature_observed_c"]
            dates = pd.to_datetime(["2018-02-15 10:00", "2019-02-15 10:00"])
            selected = pd.DatetimeIndex([d + pd.Timedelta(days=offset)
                                        for d in dates for offset in range(-(width // 2), width // 2 + 1)])
            expected = known.reindex(selected).mean()
            self.assertAlmostEqual(analogue.train_quality.loc[stamp, "temperature_input_c"], expected)
            self.assertAlmostEqual(analogue.future_metadata.loc["2020-02-15 10:00", "temperature_input_c"], expected)
            self.assertTrue(analogue.future_metadata.loc["2020-02-15 10:00", "temperature_estimated"])
            self.assertAlmostEqual(climate.future_metadata.loc["2020-02-15 10:00", "temperature_input_c"], expected)
            assert_frame_equal(analogue.X_train, climate.X_train)
            poisoned = changed.copy()
            poisoned.loc["2020":, ["TOTALDEMAND", "temperature_observed_c"]] = 999999
            other = build_fold(poisoned, self.split, self.terms, "climatology", config)
            assert_frame_equal(climate.X_train, other.X_train)
            assert_frame_equal(climate.X_future, other.X_future)

    def test_incomplete_forecast_or_donor_fails(self):
        with self.assertRaisesRegex(ValueError, "fully covered"):
            build_fold(self.data.iloc[:-1], self.split, self.terms)
        short_train = dict(self.split, train_start="2019-04-01")
        with self.assertRaisesRegex(ValueError, "previous complete"):
            build_fold(self.data, short_train, self.terms)

    def test_development_cannot_use_reserved_year(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            build_fold(self.data, dict(self.split, role="development"), self.terms)

    def test_invalid_mode_and_midmonth_origin_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown weather"):
            build_fold(self.data, self.split, self.terms, "future_actuals")
        with self.assertRaisesRegex(ValueError, "first-of-month"):
            build_fold(self.data, dict(self.split, origin="2020-01-02"), self.terms)

    def test_degree_features(self):
        changed = self.data.copy()
        changed.loc["2019-01-01 12:00", "temperature_observed_c"] = 25
        changed.loc["2019-01-01 12:30", "temperature_observed_c"] = 12
        fold = build_fold(changed, self.split, self.terms)
        self.assertEqual(fold.X_future.loc["2020-01-01 12:00", "cooling_degrees_18c"], 7)
        self.assertEqual(fold.X_future.loc["2020-01-01 12:00", "heating_degrees_18c"], 0)
        self.assertEqual(fold.X_future.loc["2020-01-01 12:30", "heating_degrees_18c"], 6)

    def test_dst_changes_local_features_without_changing_row_identity(self):
        index = pd.to_datetime(["2020-10-04 01:30", "2020-10-04 02:00"])
        features = calendar_features(index, self.terms)
        self.assertEqual(list(features.hour), [1, 3])
        self.assertEqual(list(features.utc_offset_hours), [10, 11])
        self.assertTrue(features.index.equals(index))

    def test_local_date_controls_holidays(self):
        features = calendar_features(pd.to_datetime(["2019-12-31 23:00"]), self.terms)
        self.assertEqual(features.is_public_holiday.iloc[0], 1)  # Jan 1 in Sydney

    def test_school_breaks_and_term_weekends(self):
        stamps = pd.to_datetime(["2020-01-27 12:00", "2020-01-28 12:00",
                                  "2020-02-01 12:00", "2020-04-11 12:00"])
        features = calendar_features(stamps, self.terms)
        self.assertEqual(list(features.is_school_holiday), [1, 0, 0, 1])
        self.assertEqual(list(features.is_weekend), [0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "cover"):
            calendar_features(pd.to_datetime(["2022-02-01"]), self.terms)

    def test_public_holiday_historical_exceptions(self):
        stamps = pd.to_datetime(["2010-04-25 12:00", "2010-04-26 12:00",
                                  "2010-08-02 12:00", "2010-12-26 12:00",
                                  "2010-12-28 12:00", "2011-04-26 12:00",
                                  "2014-01-26 12:00", "2014-01-27 12:00",
                                  "2020-04-10 12:00", "2020-08-03 12:00",
                                  "2020-12-28 12:00"])
        features = calendar_features(stamps, self.terms)
        self.assertEqual(list(features.is_public_holiday), [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1])

    def test_evaluation_strata_and_prediction_alignment(self):
        evaluation = evaluation_frame(self.data, self.fold)
        self.assertEqual(set(evaluation.season), {"summer", "autumn", "winter", "spring"})
        self.assertIn("y_true", evaluation)
        self.assertNotIn("y_true", self.fold.X_future)
        predictions = pd.Series(7000., index=self.fold.X_future.index)
        result = predictions_frame(self.fold, predictions, "example")
        self.assertEqual(len(result), len(evaluation))
        with self.assertRaisesRegex(ValueError, "index"):
            predictions_frame(self.fold, predictions.iloc[::-1], "example")
        with self.assertRaisesRegex(ValueError, "finite"):
            predictions_frame(self.fold, [np.nan] * len(predictions), "example")

    def test_export_separates_truth_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = export_fold(self.fold, directory)
            self.assertTrue((folder / "X_future.csv").exists())
            self.assertFalse((folder / "y_future.csv").exists())
            columns = pd.read_csv(folder / "X_future.csv", nrows=0).columns
            self.assertNotIn("TOTALDEMAND", columns)
            with self.assertRaises(FileExistsError):
                export_fold(self.fold, directory)


class ClimatologyTests(unittest.TestCase):
    def test_three_and_five_day_windows_pool_valid_readings_at_same_time(self):
        known = pd.Series([100., 10., 20., np.nan, 200., 30., 40., 50., 999.],
                          index=pd.to_datetime([
                              "2017-02-13 10:00", "2017-02-14 10:00", "2017-02-15 10:00",
                              "2017-02-16 10:00", "2017-02-17 10:00", "2018-02-14 10:00",
                              "2018-02-15 10:00", "2018-02-16 10:00", "2018-02-15 10:30",
                          ]))
        target = pd.to_datetime(["2019-02-15 10:00"])
        self.assertEqual(_climatology(known, target).iloc[0], 30.)
        self.assertAlmostEqual(_climatology(known, target, FeatureConfig(climatology_window_days=5)).iloc[0],
                               450 / 7)

    def test_windows_cross_month_and_year_boundaries(self):
        known = pd.Series([10., 20., 30., 100., 40., 50., 60.], index=pd.to_datetime([
            "2017-12-31 10:00", "2018-01-01 10:00", "2018-01-02 10:00", "2018-01-03 10:00",
            "2018-04-30 10:00", "2018-05-01 10:00", "2018-05-02 10:00",
        ]))
        actual = _climatology(known, pd.to_datetime(["2019-01-01 10:00", "2019-05-01 10:00"]))
        np.testing.assert_allclose(actual, [20., 50.])

    def test_february_29_centres_on_february_28_in_nonleap_years(self):
        known = pd.Series([10., 20., 30., 40., 50., 60.], index=pd.to_datetime([
            "2016-02-28 10:00", "2016-02-29 10:00", "2016-03-01 10:00",
            "2017-02-27 10:00", "2017-02-28 10:00", "2017-03-01 10:00",
        ]))
        self.assertEqual(_climatology(known, pd.to_datetime(["2020-02-29 10:00"])).iloc[0], 35.)

    def test_empty_window_uses_only_available_training_mean(self):
        known = pd.Series([10., np.nan, 30.], index=pd.to_datetime([
            "2018-01-01 10:00", "2018-01-02 10:00", "2018-01-03 10:00",
        ]))
        self.assertEqual(_climatology(known, pd.to_datetime(["2019-07-01 10:00"])).iloc[0], 20.)

    def test_invalid_window_widths(self):
        for width in (0, -1, 2, 4, 3.0, True, 367):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "odd integer"):
                FeatureConfig(climatology_window_days=width)


class RealDataTests(unittest.TestCase):
    def test_source_data_and_known_weather_gap(self):
        data = load_prepared_data()
        self.assertEqual(len(data), 196513)
        self.assertEqual(int(data.temperature_observed_c.isna().sum()), 341)
        self.assertEqual(data.temperature_offset_minutes.abs().max(), 2700)
        self.assertTrue(pd.isna(data.loc["2016-07-17 13:30", "temperature_observed_c"]))
        self.assertEqual(data.TOTALDEMAND.iloc[0], 8038.)
        self.assertEqual(len(data.attrs["source_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
