"""Task 11: shared, origin-aware inputs for NSW demand forecasting.

An origin is the FIRST timestamp to predict. Training rows and weather source
timestamps must precede it. No model is fitted and no files are written on import.
See task11_handoff.md for assumptions and the submitted-plan decision log.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import holidays
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET = "TOTALDEMAND"
STEP = pd.Timedelta(minutes=30)
PROTOCOL_VERSION = "task11-v2"


@dataclass(frozen=True)
class FeatureConfig:
    # Source files carry no timezone metadata. These are explicit assumptions,
    # pending confirmation against the course data brief, not verified facts.
    demand_timezone: str = "Etc/GMT-10"  # fixed UTC+10 (NEM market time)
    temperature_timezone: str = "Etc/GMT-10"
    calendar_timezone: str = "Australia/Sydney"
    temperature_tolerance_minutes: int = 30
    climatology_window_days: int = 3  # Total centred width: 3 means previous/current/next day.
    base_temperature_c: float = 18.0
    daily_harmonics: int = 3
    weekly_harmonics: int = 2
    annual_harmonics: int = 3

    def __post_init__(self):
        width = self.climatology_window_days
        if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 365 or width % 2 == 0:
            raise ValueError("Climatology window must be an odd integer from 1 to 365 days.")
        if self.temperature_tolerance_minutes < 0:
            raise ValueError("Temperature tolerance cannot be negative.")
        if not np.isfinite(self.base_temperature_c):
            raise ValueError("Temperature base must be finite.")
        for n in (self.daily_harmonics, self.weekly_harmonics, self.annual_harmonics):
            if not isinstance(n, int) or n < 0:
                raise ValueError("Harmonic counts must be nonnegative integers.")


@dataclass
class ForecastFold:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_future: pd.DataFrame
    future_metadata: pd.DataFrame
    train_quality: pd.DataFrame
    metadata: dict


def _index(values) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values, name="DATETIME")
    if index.tz is not None or index.hasnans or index.has_duplicates:
        raise ValueError("Use unique, non-missing, naive source DATETIME labels.")
    if not index.is_monotonic_increasing:
        raise ValueError("DATETIME must be sorted.")
    return index


def _local_time(index, config):
    return index.tz_localize(
        config.demand_timezone, ambiguous="raise", nonexistent="raise"
    ).tz_convert(config.calendar_timezone)


def load_prepared_data(root=ROOT, config=None) -> pd.DataFrame:
    """Read Task 10 output and recover temperature match provenance from raw ZIP.

    Original CSV/ZIP files are never changed. Demand remains on the original
    timeline. Matches outside tolerance become NaN for fold-local imputation.
    AEMO forecast columns are deliberately not read into the shared table.
    """
    config = config or FeatureConfig()
    folder = Path(root) / "data" / "NSW"
    data = pd.read_csv(folder / "nsw_cleaned.csv", usecols=["DATETIME", TARGET])
    data["DATETIME"] = pd.to_datetime(data["DATETIME"], format="ISO8601")
    index = _index(data["DATETIME"])
    if len(index) < 2 or not (index[1:] - index[:-1] == STEP).all():
        raise ValueError("Demand must have a complete, regular half-hourly timeline.")
    if not np.isfinite(data[TARGET]).all() or not (data[TARGET] > 0).all():
        raise ValueError("Demand must be finite and positive.")
    weather = pd.read_csv(folder / "temperature_nsw.csv.zip")
    if set(weather["LOCATION"].dropna()) != {"Bankstown"}:
        raise ValueError("Expected the single Bankstown weather station.")
    weather = weather.drop_duplicates().dropna(subset=["TEMPERATURE"])
    raw_times = pd.DatetimeIndex(pd.to_datetime(weather["DATETIME"], dayfirst=True))
    weather["temperature_source_time"] = raw_times.tz_localize(
        config.temperature_timezone, ambiguous="raise", nonexistent="raise"
    ).tz_convert(config.demand_timezone).tz_localize(None)
    weather = weather[["temperature_source_time", "TEMPERATURE"]].sort_values(
        "temperature_source_time"
    )
    if weather["temperature_source_time"].duplicated().any():
        raise ValueError("Conflicting temperature readings at the same timestamp.")
    if not np.isfinite(weather["TEMPERATURE"]).all():
        raise ValueError("Temperature must be finite.")
    joined = pd.merge_asof(
        data, weather, left_on="DATETIME", right_on="temperature_source_time",
        direction="nearest",
    ).set_index("DATETIME")
    joined["temperature_offset_minutes"] = (
        joined["temperature_source_time"] - joined.index
    ).dt.total_seconds() / 60
    valid = joined["temperature_offset_minutes"].abs().le(
        config.temperature_tolerance_minutes
    )
    joined["temperature_observed_c"] = joined["TEMPERATURE"].where(valid)
    joined = joined.drop(columns="TEMPERATURE")
    joined.attrs["config"] = asdict(config)
    joined.attrs["source_sha256"] = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("nsw_cleaned.csv", "temperature_nsw.csv.zip")
    }
    return joined


def load_school_terms(root=ROOT):
    terms = pd.read_csv(Path(root) / "data" / "NSW" / "calendar" / "school_terms.csv")
    for col in ("start_date", "end_date"):
        terms[col] = pd.to_datetime(terms[col], format="%Y-%m-%d")
    if terms.empty or terms[["year", "term"]].duplicated().any():
        raise ValueError("School calendar must have one row per year and term.")
    if not terms.groupby("year")["term"].apply(lambda s: set(s) == {1, 2, 3, 4}).all():
        raise ValueError("Every calendar year needs all four terms.")
    if not ((terms.start_date <= terms.end_date)
            & (terms.start_date.dt.year == terms.year)
            & (terms.end_date.dt.year == terms.year)).all():
        raise ValueError("Invalid term boundaries.")
    ordered = terms.sort_values("start_date")
    if (ordered.start_date.iloc[1:].to_numpy() <= ordered.end_date.iloc[:-1].to_numpy()).any():
        raise ValueError("School terms overlap.")
    return ordered


def public_holiday_calendar(years):
    """NSW statewide dates, with sourced 2010/2011 library corrections.

    See calendar/README.md: holidays 0.104 misses special proclamations and
    includes the 2010 bank-only day and substituted Sundays.
    """
    years = sorted(set(years))
    result = dict(holidays.country_holidays("AU", subdiv="NSW", years=years, observed=True))
    result = {day: name for day, name in result.items() if name != "Bank Holiday"}
    if 2010 in years:
        for day in ("2010-04-25", "2010-12-26"):
            result.pop(pd.Timestamp(day).date(), None)
        result[pd.Timestamp("2010-12-28").date()] = "Additional public holiday"
    if 2011 in years:
        result[pd.Timestamp("2011-04-25").date()] = "ANZAC Day"
        result[pd.Timestamp("2011-04-26").date()] = "Easter Monday (substituted)"
    return result


def calendar_features(index, terms, config=None):
    """Deterministic features, safe to calculate for dates beyond the origin."""
    config = config or FeatureConfig()
    index = _index(index)
    local = _local_time(index, config)
    dates = local.tz_localize(None).normalize()
    if not set(dates.year).issubset(set(terms.year)):
        raise ValueError("School calendar does not cover every requested year.")
    slot = local.hour * 2 + local.minute // 30
    out = pd.DataFrame({
        "half_hour": slot, "hour": local.hour, "day_of_week": local.dayofweek,
        "month": local.month, "day_of_year": local.dayofyear,
        "is_weekend": (local.dayofweek >= 5).astype(int),
        "utc_offset_hours": [t.utcoffset().total_seconds() / 3600 for t in local],
    }, index=index)
    public = public_holiday_calendar(dates.year)
    out["is_public_holiday"] = [int(d.date() in public) for d in dates]
    in_term = np.zeros(len(index), dtype=bool)
    for term in terms.itertuples():
        in_term |= (dates >= term.start_date) & (dates <= term.end_date)
    # Includes weekends within vacations; ordinary term-time weekends are not
    # school holidays. Published term boundaries include staff development days.
    out["is_school_holiday"] = (~in_term).astype(int)
    elapsed_days = (index - pd.Timestamp("2010-01-01")) / pd.Timedelta(days=1)
    out["trend_years"] = elapsed_days / 365.2425
    phases = {
        "daily": (slot / 48, config.daily_harmonics),
        "weekly": ((local.dayofweek + slot / 48) / 7, config.weekly_harmonics),
        "annual": ((local.dayofyear - 1 + slot / 48) / np.where(local.is_leap_year, 366, 365),
                   config.annual_harmonics),
    }
    for period, (phase, count) in phases.items():
        for k in range(1, count + 1):
            out[f"{period}_sin_{k}"] = np.sin(2 * np.pi * k * phase)
            out[f"{period}_cos_{k}"] = np.cos(2 * np.pi * k * phase)
    return out.astype(float)


def default_splits():
    """Eight development origins, then two reserved complete-year tests."""
    rows = []
    for role, dates in (
        ("development", pd.date_range("2016-01-01", "2017-10-01", freq="QS")),
        ("final_test", pd.to_datetime(["2019-01-01", "2020-01-01"])),
    ):
        for origin in dates:
            rows.append({"fold_id": f"{role}_{origin:%Y%m%d}", "role": role,
                         "train_start": pd.Timestamp("2010-01-01"), "origin": origin,
                         "forecast_end_exclusive": origin + pd.DateOffset(months=12)})
    return pd.DataFrame(rows)


def _climatology(known, future_index, config=None):
    """Average valid training readings in a centred calendar-date window.

    Match the same source-clock half-hour in each training year. Feb 29 uses
    Feb 28 as its centre in non-leap years. Windows cross month/year boundaries;
    only timestamps present in known contribute, and no observation is counted
    twice. Each valid reading has equal weight. Empty windows use known.mean().
    """
    config = config or FeatureConfig()
    if known.notna().sum() == 0:
        raise ValueError("No reliable training temperatures available.")
    half_width = config.climatology_window_days // 2
    overall_mean = known.mean()
    # One row per historical date, one column per half-hour. Missing readings
    # stay missing here so filled values never feed back into an average.
    daily = pd.Series(known.to_numpy(), index=pd.MultiIndex.from_arrays([
        known.index.normalize(), known.index.hour * 2 + known.index.minute // 30,
    ])).unstack().reindex(columns=range(48))
    years = range(known.index.year.min() - 1, known.index.year.max() + 2)
    target_keys = future_index.month * 100 + future_index.day
    averages = {}
    for key in np.unique(target_keys):
        month, day = divmod(int(key), 100)
        candidates = set()
        for year in years:
            try:
                centre = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:  # Feb 29 in a non-leap training year
                centre = pd.Timestamp(year=year, month=2, day=28)
            candidates.update(pd.date_range(
                centre - pd.Timedelta(days=half_width), periods=config.climatology_window_days,
            ))
        readings = daily.reindex(pd.DatetimeIndex(sorted(candidates)))
        averages[key] = readings.mean().fillna(overall_mean).to_numpy()
    table = pd.DataFrame.from_dict(averages, orient="index")
    slots = future_index.hour * 2 + future_index.minute // 30
    values = table.loc[target_keys].to_numpy()[np.arange(len(future_index)), slots]
    return pd.Series(values, index=future_index, name="temperature_input_c")


def _with_temperature(features, temperature, config):
    out = features.copy()
    out["cooling_degrees_18c"] = (temperature - config.base_temperature_c).clip(lower=0)
    out["heating_degrees_18c"] = (config.base_temperature_c - temperature).clip(lower=0)
    # A custom base is allowed for later T13 work; keep column names accurate.
    if config.base_temperature_c != 18:
        out = out.rename(columns={"cooling_degrees_18c": "cooling_degrees",
                                  "heating_degrees_18c": "heating_degrees"})
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite model inputs.")
    return out


def build_fold(data, split, terms, weather_mode="analogue", config=None):
    """Return training inputs/labels and future inputs, never future demand.

    observed_diagnostic deliberately uses future observed weather, but neither
    mode fits transformations on future data. Training inputs are identical.
    """
    config = config or FeatureConfig()
    if data.attrs.get("config", asdict(config)) != asdict(config):
        raise ValueError("Use the same configuration when loading data and building folds.")
    if weather_mode not in {"analogue", "observed_diagnostic", "climatology"}:
        raise ValueError("Unknown weather mode.")
    _index(data.index)
    origin = pd.Timestamp(split["origin"])
    start = pd.Timestamp(split["train_start"])
    end = pd.Timestamp(split["forecast_end_exclusive"])
    if origin != origin.normalize().replace(day=1) or origin.tz is not None:
        raise ValueError("Origins must be naive first-of-month midnight timestamps.")
    if not start < origin < end or end > origin + pd.DateOffset(months=12):
        raise ValueError("Require nonempty training and a forecast of at most 12 months.")
    if end != end.normalize().replace(day=1):
        raise ValueError("Forecast end must be first-of-month midnight (exclusive).")
    if split.get("role") == "development" and end > pd.Timestamp("2019-01-01"):
        raise ValueError("Development cannot consume reserved 2019/2020 outcomes.")
    train_index = pd.date_range(start, origin, freq="30min", inclusive="left", name="DATETIME")
    future_index = pd.date_range(origin, end, freq="30min", inclusive="left", name="DATETIME")
    if not train_index.isin(data.index).all() or not future_index.isin(data.index).all():
        raise ValueError("Requested fold is not fully covered by the dataset.")
    train = data.loc[train_index]
    if not np.isfinite(train[TARGET]).all() or not (train[TARGET] > 0).all():
        raise ValueError("Training demand must be finite and positive.")
    # Mask nearest readings that only became available at/after the origin.
    known = train.temperature_observed_c.where(train.temperature_source_time < origin)
    train_temperature = known.fillna(_climatology(known, train_index, config))
    donor_year = None
    source_times = pd.Series(pd.NaT, index=future_index, dtype="datetime64[ns]")
    leap_mapped = pd.Series(False, index=future_index)
    fallback = pd.Series(False, index=future_index)
    if weather_mode == "analogue":
        donor_year = origin.year - 1
        donor_index = pd.date_range(f"{donor_year}-01-01", f"{donor_year + 1}-01-01",
                                   inclusive="left", freq="30min")
        if not donor_index.isin(train_index).all():
            raise ValueError("Analogue mode requires the previous complete calendar year in training.")
        # Fixed donor chosen before examining forecast-year weather or demand.
        mapped = []
        for t in future_index:
            try:
                mapped.append(t.replace(year=donor_year))
            except ValueError:  # Feb 29 in a non-leap donor year
                mapped.append(t.replace(year=donor_year, day=28))
        donor_dates = pd.DatetimeIndex(mapped)
        temperature = pd.Series(train_temperature.reindex(donor_dates).to_numpy(), index=future_index)
        source_times = pd.Series(train.temperature_source_time.reindex(donor_dates).to_numpy(), index=future_index)
        fallback = pd.Series(known.reindex(donor_dates).isna().to_numpy(), index=future_index)
        source_times = source_times.mask(fallback)
        leap_mapped = pd.Series((future_index.month == 2) & (future_index.day == 29)
                               & (donor_dates.day == 28), index=future_index)
    elif weather_mode == "climatology":
        temperature = _climatology(known, future_index, config)
        fallback[:] = True  # all values are training-derived estimates
    else:
        observed = data.loc[future_index, "temperature_observed_c"]
        temperature = observed.fillna(_climatology(known, future_index, config))
        fallback = observed.isna()
        source_times = data.loc[future_index, "temperature_source_time"].mask(fallback)
    future_metadata = pd.DataFrame({
        "lead_month": (future_index.year - origin.year) * 12 + future_index.month - origin.month + 1,
        "temperature_input_c": temperature,
        "temperature_source_time": source_times,
        "temperature_estimated": fallback,
        "leap_day_mapped_to_feb28": leap_mapped,
    }, index=future_index)
    metadata = {
        "protocol_version": PROTOCOL_VERSION, "fold_id": str(split["fold_id"]),
        "role": str(split["role"]), "origin": origin.isoformat(),
        "train_start": start.isoformat(), "forecast_end_exclusive": end.isoformat(),
        "weather_mode": weather_mode, "analogue_year": donor_year,
        "train_rows": len(train_index), "forecast_rows": len(future_index),
        "train_temperature_imputed_rows": int(known.isna().sum()),
        "future_temperature_estimated_rows": int(fallback.sum()),
        "config": asdict(config), "holidays_version": holidays.__version__,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("pandas", "numpy", "holidays", "tzdata")},
        "python": platform.python_version(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": data.attrs.get("source_sha256", {}),
        "school_calendar_sha256": hashlib.sha256(terms.to_csv(index=False).encode()).hexdigest(),
        "timezone_status": "source-timezone assumptions require course-data confirmation",
    }
    return ForecastFold(
        X_train=_with_temperature(calendar_features(train_index, terms, config), train_temperature, config),
        y_train=train[TARGET].copy(),
        X_future=_with_temperature(calendar_features(future_index, terms, config), temperature, config),
        future_metadata=future_metadata,
        train_quality=pd.DataFrame({"temperature_imputed": known.isna(),
                                    "temperature_input_c": train_temperature}, index=train_index),
        metadata=metadata,
    )


def evaluation_frame(data, fold):
    """Explicit scoring-only access: future demand and observed-weather groups.

    Never pass this table as model input. Demand bands are defined from training
    terciles; actual demand/weather are used only to assign evaluation groups.
    """
    index = fold.X_future.index
    config = FeatureConfig(**fold.metadata["config"])
    local = _local_time(index, config)
    truth = data.loc[index, [TARGET, "temperature_observed_c"]].copy()
    truth = truth.rename(columns={TARGET: "y_true"})
    truth["lead_month"] = fold.future_metadata["lead_month"]
    truth["season"] = np.array(["summer", "autumn", "winter", "spring"])[(local.month % 12) // 3]
    truth["temperature_band"] = pd.cut(
        truth.temperature_observed_c, [-np.inf, 18, 26, np.inf],
        labels=["below_18", "18_to_below_26", "26_and_above"], right=False,
    ).astype("string").fillna("unreliable_or_missing")
    low, high = fold.y_train.quantile([1 / 3, 2 / 3])
    truth["demand_band"] = np.where(truth.y_true <= low, "low",
                                     np.where(truth.y_true <= high, "medium", "high"))
    return truth


def predictions_frame(fold, predictions, model_name):
    """One record per model/origin/scenario/timestamp for Daniel's T20 harness."""
    if isinstance(predictions, pd.Series) and not predictions.index.equals(fold.X_future.index):
        raise ValueError("Prediction Series index must exactly match X_future.index.")
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(fold.X_future),) or not np.isfinite(values).all():
        raise ValueError("Supply one finite prediction per forecast timestamp.")
    result = pd.DataFrame({"DATETIME": fold.X_future.index, "y_pred": values})
    for key in ("fold_id", "origin", "weather_mode", "protocol_version"):
        result[key] = fold.metadata[key]
    result["model"] = model_name
    result["lead_month"] = fold.future_metadata.lead_month.to_numpy()
    return result


def export_fold(fold, output_dir):
    """Optional CSV handoff. Does not export future demand or overwrite a run."""
    folder = Path(output_dir) / fold.metadata["fold_id"] / fold.metadata["weather_mode"]
    folder.mkdir(parents=True, exist_ok=False)
    fold.X_train.to_csv(folder / "X_train.csv")
    fold.y_train.to_csv(folder / "y_train.csv")
    fold.X_future.to_csv(folder / "X_future.csv")
    fold.train_quality.to_csv(folder / "train_quality.csv")
    fold.future_metadata.to_csv(folder / "future_metadata.csv")
    (folder / "metadata.json").write_text(json.dumps(fold.metadata, indent=2), encoding="utf-8")
    return folder
