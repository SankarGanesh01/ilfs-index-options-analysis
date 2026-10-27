# %% [markdown]
# ==================================================
# STEP 1: DATA CLEANING AND DESCRIPTIVE STATISTICS
# ==================================================
#
# Research context:
# Indian Financial Crisis (IL&FS Default) - 2018
#
# Scope of this notebook:
# - Data loading
# - Data cleaning and validation
# - Option-level master dataset creation
# - Feature engineering required for later notebooks
# - Descriptive statistics
# - Exploratory visualisations
# - CSV and PNG output generation
#
# This notebook deliberately excludes Black-Scholes, Greeks, implied volatility,
# hedging, regression, put-call parity, and event study analysis.

# %%
# ==================================================
# 1. IMPORT LIBRARIES
# ==================================================

import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)


# %%
# ==================================================
# 2. CONFIGURATION
# ==================================================

BASE_DIR = Path.cwd()

DATA_FILES = {
    "banknifty_spot": BASE_DIR / "banknifty_spot.csv",
    "nifty_spot": BASE_DIR / "nifty_spot.csv",
    "india_vix": BASE_DIR / "india_vix.csv",
    "risk_free_rate": BASE_DIR / "risk_free_rate.xlsx",
    "derivatives": BASE_DIR / "combined_nifty_banknifty.csv",
}

OUTPUT_DIR = BASE_DIR / "Output"
FIGURES_DIR = BASE_DIR / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# %%
# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

quality_audit = {}
warnings_list = []
conversion_issues = {}
merge_diagnostics = []
EVENT_DATE = pd.Timestamp("2018-09-04")
EXECUTION_START_TIME = time.perf_counter()


def print_section(title: str) -> None:
    """Print a clear section heading for notebook logs."""
    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


def require_file(path: Path) -> None:
    """Stop early if an expected source file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def snake_case_column(column_name: object) -> str:
    """Convert inconsistent source column names into stable snake_case."""
    text = str(column_name).strip()
    text = text.replace("\u20b9", "inr")
    text = text.replace("%", "percent")
    text = text.replace("&", "and")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply snake_case column names."""
    df = df.copy()
    df.columns = [snake_case_column(col) for col in df.columns]
    return df


def initialise_audit(label: str, df: pd.DataFrame) -> None:
    """Create a data quality audit record for one dataset."""
    quality_audit[label] = {
        "rows_before": len(df),
        "columns_before": df.shape[1],
        "rows_after": len(df),
        "columns_after": df.shape[1],
        "duplicates_removed": 0,
        "columns_removed": [],
        "date_conversion_failures": {},
        "numeric_conversion_failures": {},
    }


def remove_empty_unnamed_and_duplicate_rows(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Remove fully empty columns, unnamed columns, and duplicate rows."""
    df = df.copy()
    initialise_audit(label, df)

    empty_cols = [col for col in df.columns if df[col].isna().all()]
    unnamed_cols = [
        col
        for col in df.columns
        if str(col).strip() == "" or str(col).lower().startswith("unnamed")
    ]
    drop_cols = sorted(set(empty_cols + unnamed_cols))

    if drop_cols:
        df = df.drop(columns=drop_cols)

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    quality_audit[label]["duplicates_removed"] = duplicate_count
    quality_audit[label]["columns_removed"] = drop_cols
    quality_audit[label]["rows_after"] = len(df)
    quality_audit[label]["columns_after"] = df.shape[1]

    return df


def parse_dates_with_formats(series: pd.Series, formats: list[str]) -> pd.Series:
    """Parse dates with known formats before using a conservative fallback."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    raw_text = series.astype("string").str.strip()
    non_empty = series.notna() & raw_text.notna() & (raw_text != "")

    for fmt in formats:
        remaining = non_empty & result.isna()
        if not remaining.any():
            break
        parsed = pd.to_datetime(raw_text.loc[remaining], format=fmt, errors="coerce")
        parsed_ok = parsed.notna()
        result.loc[parsed.index[parsed_ok]] = parsed.loc[parsed_ok]

    remaining = non_empty & result.isna()
    if remaining.any():
        fallback = pd.to_datetime(raw_text.loc[remaining], errors="coerce", dayfirst=False)
        fallback_ok = fallback.notna()
        result.loc[fallback.index[fallback_ok]] = fallback.loc[fallback_ok]

    return result


def coerce_date_columns(
    df: pd.DataFrame,
    label: str,
    date_columns: list[str],
    date_formats: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Convert date columns to pandas datetime and report failed conversions."""
    df = df.copy()
    date_formats = date_formats or {}

    for col in date_columns:
        if col not in df.columns:
            continue

        original = df[col].copy()
        formats = date_formats.get(col, ["%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"])
        converted = parse_dates_with_formats(original, formats)
        failures = original.notna() & converted.isna()
        failure_count = int(failures.sum())

        if failure_count:
            examples = original[failures].astype(str).drop_duplicates().head(10).tolist()
            quality_audit[label]["date_conversion_failures"][col] = failure_count
            warnings_list.append(
                f"{label}: {failure_count} values in '{col}' could not be converted to datetime. Examples: {examples}"
            )

        df[col] = converted

    return df


def clean_numeric_text(series: pd.Series) -> pd.Series:
    """Prepare numeric-looking text for conversion."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("\u20b9", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("--", "", regex=False)
    )
    return cleaned


def coerce_numeric_columns(df: pd.DataFrame, label: str, numeric_columns: list[str]) -> pd.DataFrame:
    """Convert expected numeric columns and retain examples of failed conversions."""
    df = df.copy()

    for col in numeric_columns:
        if col not in df.columns:
            continue

        original = df[col].copy()
        cleaned = clean_numeric_text(original)
        converted = pd.to_numeric(cleaned, errors="coerce")

        non_empty_original = original.notna() & cleaned.notna() & (cleaned != "")
        failures = non_empty_original & converted.isna()
        failure_count = int(failures.sum())

        if failure_count:
            examples = original[failures].astype(str).drop_duplicates().head(10).tolist()
            conversion_issues.setdefault(label, {})[col] = {
                "failed_count": failure_count,
                "examples": examples,
            }
            quality_audit[label]["numeric_conversion_failures"][col] = failure_count
            warnings_list.append(
                f"{label}: {failure_count} values in '{col}' could not be converted to numeric. Examples: {examples}"
            )

        df[col] = converted

    return df


def sort_and_reset(df: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    """Sort by available columns and reset the index."""
    available_columns = [col for col in sort_columns if col in df.columns]
    if available_columns:
        df = df.sort_values(available_columns)
    return df.reset_index(drop=True)


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return deep memory usage in megabytes."""
    return df.memory_usage(deep=True).sum() / (1024**2)


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value summary table."""
    missing = df.isna().sum().rename("missing_count").reset_index()
    missing = missing.rename(columns={"index": "column"})
    missing["missing_percent"] = (missing["missing_count"] / len(df) * 100).round(4)
    return missing.sort_values(["missing_count", "column"], ascending=[False, True])


def descriptive_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute descriptive statistics with skewness and kurtosis."""
    available = [col for col in columns if col in df.columns]
    stats = []

    for col in available:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        stats.append(
            {
                "variable": col,
                "count": int(series.count()),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "25%": series.quantile(0.25),
                "50%": series.quantile(0.50),
                "75%": series.quantile(0.75),
                "max": series.max(),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
            }
        )

    return pd.DataFrame(stats)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save and close a matplotlib figure."""
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_csv_output(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Save a CSV, falling back to a timestamped file if Windows locks the target."""
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError as exc:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback_path, index=index)
        warning = (
            f"Could not overwrite {path} because the file appears to be locked; "
            f"saved fallback output to {fallback_path}."
        )
        warnings_list.append(warning)
        print(f"WARNING: {warning}")
        return fallback_path


def report_duplicate_keys(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Report duplicate business-key observations."""
    available = [col for col in key_columns if col in df.columns]
    if not available:
        return pd.DataFrame()

    duplicate_mask = df.duplicated(subset=available, keep=False)
    duplicate_counts = (
        df.loc[duplicate_mask, available]
        .value_counts()
        .rename("duplicate_count")
        .reset_index()
        .sort_values("duplicate_count", ascending=False)
    )
    return duplicate_counts



def left_merge_with_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    merge_name: str,
    required_columns: list[str] | None = None,
    **merge_kwargs,
) -> pd.DataFrame:
    """Run a left merge, print diagnostics, and fail clearly on validation errors."""
    rows_before = len(left)
    indicator_col = f"__merge_{snake_case_column(merge_name)}"

    try:
        merged = left.merge(right, how="left", indicator=indicator_col, **merge_kwargs)
    except pd.errors.MergeError as exc:
        raise ValueError(
            f"{merge_name} merge validation failed. Review merge keys and duplicate right-side observations."
        ) from exc

    rows_after = len(merged)
    if required_columns:
        available_required = [col for col in required_columns if col in merged.columns]
        if available_required:
            matched_mask = merged[available_required].notna().any(axis=1)
        else:
            matched_mask = merged[indicator_col].eq("both")
    else:
        matched_mask = merged[indicator_col].eq("both")

    matched_rows = int(matched_mask.sum())
    unmatched_rows = int(rows_after - matched_rows)
    success_pct = (matched_rows / rows_after * 100) if rows_after else 0.0
    unmatched_pct = (unmatched_rows / rows_after * 100) if rows_after else 0.0

    diagnostic = {
        "merge_name": merge_name,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "matched_rows": matched_rows,
        "merge_success_percent": round(success_pct, 4),
        "unmatched_rows": unmatched_rows,
        "unmatched_percent": round(unmatched_pct, 4),
        "validation": merge_kwargs.get("validate", "not_specified"),
    }
    merge_diagnostics.append(diagnostic)

    print(f"\nMerge diagnostics - {merge_name}")
    print(f"Rows before merge: {rows_before:,}")
    print(f"Rows after merge: {rows_after:,}")
    print(f"Merge success percentage: {success_pct:.2f}%")
    print(f"Unmatched rows: {unmatched_rows:,}")
    print(f"Percentage unmatched: {unmatched_pct:.2f}%")

    if rows_after < rows_before:
        warning = f"{merge_name}: observations were lost during merge."
        warnings_list.append(warning)
        print(f"WARNING: {warning}")
    elif rows_after > rows_before:
        warning = f"{merge_name}: merge introduced additional rows; check key uniqueness."
        warnings_list.append(warning)
        print(f"WARNING: {warning}")

    return merged.drop(columns=[indicator_col])


def build_option_future_matches(
    options_df: pd.DataFrame,
    expiry_matched_futures_df: pd.DataFrame,
    front_month_futures_df: pd.DataFrame,
    future_columns: list[str],
) -> pd.DataFrame:
    """Match each option to same-expiry futures first, then front-month futures."""
    if "__option_row_id" not in options_df.columns:
        raise KeyError("options_df must contain '__option_row_id' before futures matching.")

    option_keys = options_df[["__option_row_id", "date", "index", "expiry"]].copy()

    # Exact expiry matching preserves the economic maturity of each option contract.
    exact_matches = left_merge_with_diagnostics(
        option_keys,
        expiry_matched_futures_df,
        "Expiry-Matched Futures Lookup",
        required_columns=["future_expiry"],
        left_on=["date", "index", "expiry"],
        right_on=["date", "index", "future_expiry"],
        validate="many_to_one",
    )

    future_lookup = exact_matches[["__option_row_id"] + future_columns].copy()
    future_lookup["future_match_type"] = pd.Series(pd.NA, index=future_lookup.index, dtype="string")

    exact_matched = exact_matches["future_expiry"].notna()
    future_lookup.loc[exact_matched, "future_match_type"] = "expiry_matched"

    needs_fallback = ~exact_matched
    if needs_fallback.any():
        # If a same-expiry future is unavailable, use the nearest non-expired
        # front-month future for the same Date/Index without dropping option rows.
        fallback_matches = left_merge_with_diagnostics(
            option_keys.loc[needs_fallback, ["__option_row_id", "date", "index"]],
            front_month_futures_df,
            "Front-Month Futures Fallback Lookup",
            required_columns=["future_expiry"],
            on=["date", "index"],
            validate="many_to_one",
        )
        fallback_indexed = fallback_matches.set_index("__option_row_id")
        fallback_target_ids = future_lookup.loc[needs_fallback, "__option_row_id"]

        for col in future_columns:
            future_lookup.loc[needs_fallback, col] = fallback_target_ids.map(fallback_indexed[col])

        fallback_matched = fallback_target_ids.map(fallback_indexed["future_expiry"]).notna()
        future_lookup.loc[fallback_matched.index[fallback_matched], "future_match_type"] = (
            "front_month_fallback"
        )
    else:
        print("\nFutures matching summary")
        print("No front-month fallback was required; every option had an expiry-matched future.")

    if len(future_lookup) != len(options_df):
        raise ValueError("Futures matching changed the number of option rows.")
    if future_lookup["__option_row_id"].duplicated().any():
        raise ValueError("Futures matching produced duplicate option row identifiers.")

    match_counts = future_lookup["future_match_type"].value_counts(dropna=False)
    unmatched_after_fallback = int(future_lookup["future_match_type"].isna().sum())

    print("\nFutures matching summary")
    print(f"Expiry-matched futures used: {int(match_counts.get('expiry_matched', 0)):,}")
    print(f"Front-month fallback futures used: {int(match_counts.get('front_month_fallback', 0)):,}")
    print(f"Unmatched after fallback: {unmatched_after_fallback:,}")

    if unmatched_after_fallback:
        warnings_list.append(
            f"{unmatched_after_fallback:,} option rows could not be matched to either expiry-matched or front-month futures."
        )

    return future_lookup


def grouped_descriptive_statistics(
    df: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    """Compute the standard descriptive-statistics table by one or more groups."""
    available_groups = [col for col in group_columns if col in df.columns]
    if not available_groups:
        raise KeyError(f"None of the grouping columns exist: {group_columns}")

    grouped_tables = []
    for group_values, group_df in df.groupby(available_groups, dropna=False):
        group_values = group_values if isinstance(group_values, tuple) else (group_values,)
        stats = descriptive_statistics(group_df, value_columns)
        for col, value in zip(available_groups, group_values):
            stats.insert(0, col, value)
        grouped_tables.append(stats)

    if not grouped_tables:
        return pd.DataFrame(columns=available_groups + ["variable"])

    return pd.concat(grouped_tables, ignore_index=True)


def calculate_trading_days_to_expiry(
    df: pd.DataFrame,
    trading_dates: pd.Series,
    start_col: str = "date",
    expiry_col: str = "expiry",
) -> pd.Series:
    """Count observed trading sessions after the trade date and up to expiry."""
    clean_dates = pd.Series(pd.to_datetime(trading_dates).dropna().unique()).sort_values()
    trading_array = clean_dates.to_numpy(dtype="datetime64[ns]")

    if len(trading_array) == 0:
        warnings_list.append("Trading-day time-to-expiry could not be calculated because the trading calendar is empty.")
        return pd.Series(np.nan, index=df.index)

    start_values = pd.to_datetime(df[start_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    expiry_values = pd.to_datetime(df[expiry_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    valid_mask = ~pd.isna(start_values) & ~pd.isna(expiry_values)

    counts = np.full(len(df), np.nan)
    start_positions = np.searchsorted(trading_array, start_values[valid_mask], side="right")
    expiry_positions = np.searchsorted(trading_array, expiry_values[valid_mask], side="right")
    counts[valid_mask] = np.maximum(expiry_positions - start_positions, 0)

    if pd.to_datetime(df[expiry_col], errors="coerce").max() > clean_dates.max():
        warnings_list.append(
            "Some option expiries extend beyond the observed trading calendar; trading_days_to_expiry counts observed sessions only."
        )

    return pd.Series(counts, index=df.index).astype("Int64")


def create_data_dictionary(df: pd.DataFrame, descriptions: dict[str, str]) -> pd.DataFrame:
    """Create a dissertation-ready data dictionary for the master dataset."""
    records = []
    row_count = len(df)

    for col in df.columns:
        non_missing = df[col].dropna()
        example_value = non_missing.iloc[0] if not non_missing.empty else np.nan
        if isinstance(example_value, pd.Timestamp):
            example_value = example_value.strftime("%Y-%m-%d")

        missing_count = int(df[col].isna().sum())
        records.append(
            {
                "Variable": col,
                "Data Type": str(df[col].dtype),
                "Description": descriptions.get(
                    col,
                    "Source or audit variable retained for traceability and later derivative-market analysis.",
                ),
                "Missing Values": missing_count,
                "Missing %": round((missing_count / row_count * 100) if row_count else 0.0, 4),
                "Example Value": example_value,
            }
        )

    return pd.DataFrame(records)


def create_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report outliers without changing the underlying observations."""
    records = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        count = int(series.count())
        if count == 0:
            records.append(
                {
                    "Variable": col,
                    "Count": 0,
                    "IQR Outliers": 0,
                    "IQR Outlier %": 0.0,
                    "Z-score Outliers": 0,
                    "Z-score Outlier %": 0.0,
                    "Minimum Value": np.nan,
                    "Maximum Value": np.nan,
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = int(((series < lower_bound) | (series > upper_bound)).sum())

        std = series.std()
        if pd.isna(std) or std == 0:
            z_score_outliers = 0
        else:
            z_scores = (series - series.mean()) / std
            z_score_outliers = int((z_scores.abs() > 3).sum())

        records.append(
            {
                "Variable": col,
                "Count": count,
                "IQR Outliers": iqr_outliers,
                "IQR Outlier %": round(iqr_outliers / count * 100, 4),
                "Z-score Outliers": z_score_outliers,
                "Z-score Outlier %": round(z_score_outliers / count * 100, 4),
                "Minimum Value": series.min(),
                "Maximum Value": series.max(),
            }
        )

    return pd.DataFrame(records).sort_values("IQR Outliers", ascending=False)


def create_contract_summary(master_df: pd.DataFrame, futures_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise option and futures contract coverage by index, expiry, and strike."""
    rows = []
    futures_counts = futures_df.groupby("index").size().to_dict()
    futures_expiry_counts = futures_df.groupby(["index", "future_expiry"]).size().to_dict()

    for index_name, index_df in master_df.groupby("index", dropna=False):
        rows.append(
            {
                "summary_level": "index_summary",
                "index": index_name,
                "expiry": pd.NaT,
                "strike": np.nan,
                "option_contracts": len(index_df),
                "call_contracts": int(index_df["option_type"].eq("CE").sum()),
                "put_contracts": int(index_df["option_type"].eq("PE").sum()),
                "futures_contracts": int(futures_counts.get(index_name, 0)),
                "average_open_interest": index_df["open_interest"].mean(),
                "average_volume": index_df["volume"].mean(),
                "average_option_price": index_df["option_price"].mean(),
            }
        )

    expiry_summary = master_df.groupby(["index", "expiry"], dropna=False).agg(
        option_contracts=("option_type", "size"),
        call_contracts=("option_type", lambda x: int((x == "CE").sum())),
        put_contracts=("option_type", lambda x: int((x == "PE").sum())),
        average_open_interest=("open_interest", "mean"),
        average_volume=("volume", "mean"),
        average_option_price=("option_price", "mean"),
    ).reset_index()
    expiry_summary["summary_level"] = "contracts_by_expiry"
    expiry_summary["strike"] = np.nan
    expiry_summary["futures_contracts"] = expiry_summary.apply(
        lambda row: futures_expiry_counts.get((row["index"], row["expiry"]), 0),
        axis=1,
    )

    strike_summary = master_df.groupby(["index", "strike"], dropna=False).agg(
        option_contracts=("option_type", "size"),
        call_contracts=("option_type", lambda x: int((x == "CE").sum())),
        put_contracts=("option_type", lambda x: int((x == "PE").sum())),
        average_open_interest=("open_interest", "mean"),
        average_volume=("volume", "mean"),
        average_option_price=("option_price", "mean"),
    ).reset_index()
    strike_summary["summary_level"] = "contracts_by_strike"
    strike_summary["expiry"] = pd.NaT
    strike_summary["futures_contracts"] = np.nan

    index_summary = pd.DataFrame(rows)
    ordered_columns = [
        "summary_level",
        "index",
        "expiry",
        "strike",
        "option_contracts",
        "call_contracts",
        "put_contracts",
        "futures_contracts",
        "average_open_interest",
        "average_volume",
        "average_option_price",
    ]
    return pd.concat(
        [index_summary, expiry_summary[ordered_columns], strike_summary[ordered_columns]],
        ignore_index=True,
    )[ordered_columns]


def create_trading_calendar_validation(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    options_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    risk_free_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare source calendars to identify gaps and source-only dates."""
    source_dates = {
        "spot": set(pd.to_datetime(spot_df["date"].dropna()).dt.normalize()),
        "futures": set(pd.to_datetime(futures_df["date"].dropna()).dt.normalize()),
        "options": set(pd.to_datetime(options_df["date"].dropna()).dt.normalize()),
        "vix": set(pd.to_datetime(vix_df["date"].dropna()).dt.normalize()),
        "risk_free_rate": set(pd.to_datetime(risk_free_df["date"].dropna()).dt.normalize()),
    }
    all_dates = set().union(*source_dates.values())
    market_dates = set().union(
        source_dates["spot"],
        source_dates["futures"],
        source_dates["options"],
        source_dates["vix"],
    )
    records = []

    if market_dates:
        expected_business_dates = set(pd.bdate_range(min(market_dates), max(market_dates)))
        categories = {
            "missing_trading_dates": expected_business_dates - market_dates,
            "extra_trading_dates": market_dates - expected_business_dates,
        }

        for source_name, dates in source_dates.items():
            other_dates = set().union(
                *[other for name, other in source_dates.items() if name != source_name]
            )
            categories[f"dates_available_only_in_{source_name}"] = dates - other_dates

        for category, dates in categories.items():
            if not dates:
                records.append(
                    {
                        "category": category,
                        "date": pd.NaT,
                        "present_in_sources": "no discrepancy",
                        "issue_count": 0,
                    }
                )
                continue

            for date_value in sorted(dates):
                present_in = [name for name, dates in source_dates.items() if date_value in dates]
                records.append(
                    {
                        "category": category,
                        "date": pd.Timestamp(date_value),
                        "present_in_sources": ", ".join(present_in) if present_in else "none",
                        "issue_count": 1,
                    }
                )

    return pd.DataFrame(
        records, columns=["category", "date", "present_in_sources", "issue_count"]
    )


def summarise_conversion_failures() -> pd.DataFrame:
    """Flatten date and numeric conversion failures into a compact audit table."""
    rows = []
    for dataset_label, audit in quality_audit.items():
        for col, count in audit.get("date_conversion_failures", {}).items():
            rows.append(
                {
                    "dataset": dataset_label,
                    "column": col,
                    "conversion_type": "date",
                    "failed_count": count,
                }
            )
        for col, count in audit.get("numeric_conversion_failures", {}).items():
            rows.append(
                {
                    "dataset": dataset_label,
                    "column": col,
                    "conversion_type": "numeric",
                    "failed_count": count,
                }
            )
    return pd.DataFrame(rows, columns=["dataset", "column", "conversion_type", "failed_count"])
# %%
# ==================================================
# 4. LOAD RAW DATA
# ==================================================

print_section("4. Load Raw Data")

for file_label, file_path in DATA_FILES.items():
    require_file(file_path)
    print(f"Found {file_label}: {file_path}")

banknifty_spot_raw = pd.read_csv(DATA_FILES["banknifty_spot"], skipinitialspace=True)
nifty_spot_raw = pd.read_csv(DATA_FILES["nifty_spot"], skipinitialspace=True)
india_vix_raw = pd.read_csv(DATA_FILES["india_vix"], skipinitialspace=True)
derivatives_raw = pd.read_csv(DATA_FILES["derivatives"], skipinitialspace=True)

try:
    risk_free_rate_raw = pd.read_excel(DATA_FILES["risk_free_rate"])
except ImportError as exc:
    raise ImportError(
        "pandas needs an Excel engine such as openpyxl to read risk_free_rate.xlsx. "
        "Install/open the notebook in an environment with Excel support."
    ) from exc

print("\nRaw dataset dimensions:")
print(f"Bank NIFTY spot: {banknifty_spot_raw.shape}")
print(f"NIFTY spot: {nifty_spot_raw.shape}")
print(f"India VIX: {india_vix_raw.shape}")
print(f"Risk-free rate: {risk_free_rate_raw.shape}")
print(f"Derivatives: {derivatives_raw.shape}")


# %%
# ==================================================
# 5. STANDARDISE COLUMN NAMES AND CLEAN BASIC ISSUES
# ==================================================

print_section("5. Standardise Columns and Clean Basic Issues")

banknifty_spot = standardise_columns(banknifty_spot_raw)
nifty_spot = standardise_columns(nifty_spot_raw)
india_vix = standardise_columns(india_vix_raw)
risk_free_rate = standardise_columns(risk_free_rate_raw)
derivatives = standardise_columns(derivatives_raw)

banknifty_spot = remove_empty_unnamed_and_duplicate_rows(banknifty_spot, "banknifty_spot")
nifty_spot = remove_empty_unnamed_and_duplicate_rows(nifty_spot, "nifty_spot")
india_vix = remove_empty_unnamed_and_duplicate_rows(india_vix, "india_vix")
risk_free_rate = remove_empty_unnamed_and_duplicate_rows(risk_free_rate, "risk_free_rate")
derivatives = remove_empty_unnamed_and_duplicate_rows(derivatives, "derivatives")

print("Cleaned column names:")
for label, df in {
    "banknifty_spot": banknifty_spot,
    "nifty_spot": nifty_spot,
    "india_vix": india_vix,
    "risk_free_rate": risk_free_rate,
    "derivatives": derivatives,
}.items():
    print(f"{label}: {list(df.columns)}")


# %%
# ==================================================
# 6. DATASET-SPECIFIC CLEANING
# ==================================================

print_section("6. Dataset-Specific Cleaning")

# Spot datasets.
spot_numeric_columns = ["open", "high", "low", "close", "shares_traded", "turnover_inr_cr"]

banknifty_spot = coerce_date_columns(banknifty_spot, "banknifty_spot", ["date"])
banknifty_spot = coerce_numeric_columns(banknifty_spot, "banknifty_spot", spot_numeric_columns)
banknifty_spot["index"] = "BANKNIFTY"

nifty_spot = coerce_date_columns(nifty_spot, "nifty_spot", ["date"])
nifty_spot = coerce_numeric_columns(nifty_spot, "nifty_spot", spot_numeric_columns)
nifty_spot["index"] = "NIFTY"

spot_all = pd.concat([nifty_spot, banknifty_spot], ignore_index=True)
spot_all = spot_all.rename(
    columns={
        "open": "spot_open",
        "high": "spot_high",
        "low": "spot_low",
        "close": "spot_price",
        "turnover_inr_cr": "spot_turnover_inr_cr",
    }
)
spot_all = sort_and_reset(spot_all, ["index", "date"])

spot_all["daily_return_spot"] = spot_all.groupby("index")["spot_price"].pct_change()
spot_all["log_return_spot"] = spot_all.groupby("index")["spot_price"].transform(
    lambda x: np.log(x).diff()
)
spot_all["rolling_20d_volatility"] = (
    spot_all.groupby("index")["daily_return_spot"]
    .transform(lambda x: x.rolling(window=20, min_periods=10).std() * np.sqrt(252))
)


# India VIX dataset.
if "date" not in india_vix.columns:
    raise KeyError("India VIX file must contain a date column after standardisation.")

# Remove the malformed row containing ^INDIAVIX.
bad_vix_row = india_vix.astype("string").apply(
    lambda col: col.str.contains(r"\^INDIAVIX", na=False)
).any(axis=1)
empty_vix_date = india_vix["date"].isna() | (india_vix["date"].astype("string").str.strip() == "")
removed_vix_rows = int((bad_vix_row | empty_vix_date).sum())
if removed_vix_rows:
    print(f"Removed {removed_vix_rows} malformed India VIX row(s).")
india_vix = india_vix.loc[~(bad_vix_row | empty_vix_date)].copy()
quality_audit["india_vix"]["rows_after"] = len(india_vix)

vix_numeric_columns = ["adj_close", "close", "high", "low", "open", "volume"]
india_vix = coerce_date_columns(india_vix, "india_vix", ["date"])
india_vix = coerce_numeric_columns(india_vix, "india_vix", vix_numeric_columns)
india_vix = india_vix.rename(
    columns={
        "open": "india_vix_open",
        "high": "india_vix_high",
        "low": "india_vix_low",
        "close": "india_vix",
        "adj_close": "india_vix_adj_close",
        "volume": "india_vix_volume",
    }
)
india_vix = sort_and_reset(india_vix, ["date"])


# Risk-free-rate dataset.
risk_free_rate = risk_free_rate.rename(
    columns={
        "date_of_auction": "date",
        "auction_date": "date",
        "implicit_yield": "risk_free_rate_percent",
        "implicit_yield_at_cut_off_price_percent": "risk_free_rate_percent",
    }
)

if "date" not in risk_free_rate.columns or "risk_free_rate_percent" not in risk_free_rate.columns:
    raise KeyError(
        "risk_free_rate.xlsx must contain an auction date column and an implicit yield column."
    )

risk_free_rate = coerce_date_columns(risk_free_rate, "risk_free_rate", ["date"])
risk_free_rate = coerce_numeric_columns(
    risk_free_rate, "risk_free_rate", ["risk_free_rate_percent"]
)
risk_free_rate = risk_free_rate.dropna(subset=["date"]).copy()
quality_audit["risk_free_rate"]["rows_after"] = len(risk_free_rate)
risk_free_rate["risk_free_rate"] = risk_free_rate["risk_free_rate_percent"] / 100.0
risk_free_rate = sort_and_reset(risk_free_rate, ["date"])


# Derivatives dataset.
derivatives = derivatives.rename(
    columns={
        "instrument": "instrument",
        "symbol": "index",
        "expiry_dt": "expiry",
        "strike_pr": "strike",
        "option_typ": "option_type",
        "settle_pr": "settlement_price",
        "contracts": "volume",
        "val_inlakh": "turnover",
        "open_int": "open_interest",
        "chg_in_oi": "change_in_oi",
        "timestamp": "date",
    }
)

required_derivative_columns = [
    "instrument",
    "index",
    "expiry",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "settlement_price",
    "volume",
    "turnover",
    "open_interest",
    "change_in_oi",
    "date",
]
missing_derivative_columns = [
    col for col in required_derivative_columns if col not in derivatives.columns
]
if missing_derivative_columns:
    raise KeyError(f"Derivative file is missing columns: {missing_derivative_columns}")

derivative_numeric_columns = [
    "strike",
    "open",
    "high",
    "low",
    "close",
    "settlement_price",
    "volume",
    "turnover",
    "open_interest",
    "change_in_oi",
]
derivatives = coerce_date_columns(derivatives, "derivatives", ["date", "expiry"])
derivatives = coerce_numeric_columns(
    derivatives, "derivatives", derivative_numeric_columns
)
derivatives["index"] = derivatives["index"].astype("string").str.upper().str.strip()
derivatives["instrument"] = derivatives["instrument"].astype("string").str.upper().str.strip()
derivatives["option_type"] = derivatives["option_type"].astype("string").str.upper().str.strip()
derivatives = sort_and_reset(
    derivatives, ["date", "index", "instrument", "expiry", "strike", "option_type"]
)

print("Dataset-specific cleaning complete.")


# %%
# ==================================================
# 7. CREATE TRADING CALENDAR, DAILY VIX, AND DAILY RISK-FREE RATE
# ==================================================

print_section("7. Create Trading Calendar, Daily VIX, and Daily Risk-Free Rate")

trading_dates = (
    pd.Series(
        pd.concat(
            [
                derivatives["date"].dropna(),
                spot_all["date"].dropna(),
                india_vix["date"].dropna(),
            ],
            ignore_index=True,
        ).unique()
    )
    .sort_values()
    .reset_index(drop=True)
)
trading_calendar = pd.DataFrame({"date": trading_dates})

daily_vix = pd.merge_asof(
    trading_calendar.sort_values("date"),
    india_vix[
        [
            "date",
            "india_vix",
            "india_vix_open",
            "india_vix_high",
            "india_vix_low",
            "india_vix_adj_close",
        ]
    ].sort_values("date"),
    on="date",
    direction="backward",
)
if daily_vix["india_vix"].isna().any():
    daily_vix[
        [
            "india_vix",
            "india_vix_open",
            "india_vix_high",
            "india_vix_low",
            "india_vix_adj_close",
        ]
    ] = daily_vix[
        [
            "india_vix",
            "india_vix_open",
            "india_vix_high",
            "india_vix_low",
            "india_vix_adj_close",
        ]
    ].bfill()
    warnings_list.append(
        "Some early India VIX observations were backfilled because no earlier VIX value was available."
    )

daily_risk_free_rate = pd.merge_asof(
    trading_calendar.sort_values("date"),
    risk_free_rate[["date", "risk_free_rate_percent", "risk_free_rate"]].sort_values("date"),
    on="date",
    direction="backward",
)
if daily_risk_free_rate["risk_free_rate"].isna().any():
    daily_risk_free_rate[["risk_free_rate_percent", "risk_free_rate"]] = daily_risk_free_rate[
        ["risk_free_rate_percent", "risk_free_rate"]
    ].bfill()
    warnings_list.append(
        "Trading dates before the first available auction were backfilled with the first risk-free-rate observation."
    )

print(f"Trading calendar dates: {len(trading_calendar):,}")
print(
    "Daily risk-free-rate date range: "
    f"{daily_risk_free_rate['date'].min().date()} to {daily_risk_free_rate['date'].max().date()}"
)


# %%
# ==================================================
# 8. SPLIT OPTIONS AND FUTURES
# ==================================================

print_section("8. Split Options and Futures")

options = derivatives.loc[derivatives["instrument"].eq("OPTIDX")].copy()
futures = derivatives.loc[derivatives["instrument"].eq("FUTIDX")].copy()

options = options.rename(
    columns={
        "open": "option_open",
        "high": "option_high",
        "low": "option_low",
        "close": "option_close",
        "settlement_price": "option_settlement_price",
    }
)
options["option_price"] = options["option_settlement_price"].where(
    options["option_settlement_price"].notna(), options["option_close"]
)

# A stable row identifier lets the notebook merge futures data without changing
# the option-level grain of the master dataset.
options = options.reset_index(drop=True)
options["__option_row_id"] = np.arange(len(options))

futures = futures.rename(
    columns={
        "open": "future_open",
        "high": "future_high",
        "low": "future_low",
        "close": "future_close",
        "settlement_price": "future_settlement_price",
        "expiry": "future_expiry",
    }
)
futures["future_price"] = futures["future_settlement_price"].where(
    futures["future_settlement_price"].notna(), futures["future_close"]
)
futures["future_days_to_expiry"] = (futures["future_expiry"] - futures["date"]).dt.days

future_match_columns = [
    "future_expiry",
    "future_price",
    "future_open",
    "future_high",
    "future_low",
    "future_close",
    "future_settlement_price",
    "future_days_to_expiry",
]

# Use only non-expired futures when selecting an exact or front-month contract.
eligible_futures = futures.loc[futures["future_days_to_expiry"].ge(0)].copy()

# One same-expiry futures observation per Date/Index/Expiry supports maturity-matched
# option rows and prevents accidental duplication during the master merge.
expiry_matched_futures = (
    eligible_futures.sort_values(["date", "index", "future_expiry"])
    .drop_duplicates(subset=["date", "index", "future_expiry"], keep="first")
    .reset_index(drop=True)
)
expiry_matched_futures = expiry_matched_futures[["date", "index"] + future_match_columns]

# The front-month table is retained as a fallback for option expiries that do not
# have an exact futures maturity on the same trading date.
front_month_futures = (
    eligible_futures.sort_values(["date", "index", "future_days_to_expiry", "future_expiry"])
    .drop_duplicates(subset=["date", "index"], keep="first")
    .reset_index(drop=True)
)
front_month_futures = front_month_futures[["date", "index"] + future_match_columns]

future_match_lookup = build_option_future_matches(
    options,
    expiry_matched_futures,
    front_month_futures,
    future_match_columns,
)

print(f"Option rows: {len(options):,}")
print(f"Futures rows: {len(futures):,}")
print(f"Expiry-matched futures rows: {len(expiry_matched_futures):,}")
print(f"Front-month futures rows: {len(front_month_futures):,}")


# %%
# ==================================================
# 9. CREATE OPTION-LEVEL MASTER DATASET
# ==================================================

print_section("9. Create Option-Level Master Dataset")

spot_merge = spot_all[
    [
        "date",
        "index",
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_price",
        "shares_traded",
        "spot_turnover_inr_cr",
        "daily_return_spot",
        "log_return_spot",
        "rolling_20d_volatility",
    ]
].copy()

# Spot, futures, VIX, and risk-free-rate data are merged separately so each
# market input has its own row-count and match-rate diagnostics.
master_dataset = left_merge_with_diagnostics(
    options,
    spot_merge,
    "Spot Data",
    required_columns=["spot_price"],
    on=["date", "index"],
    validate="many_to_one",
)
master_dataset = left_merge_with_diagnostics(
    master_dataset,
    future_match_lookup,
    "Expiry-Matched or Front-Month Futures",
    required_columns=["future_expiry"],
    on="__option_row_id",
    validate="one_to_one",
)
master_dataset = left_merge_with_diagnostics(
    master_dataset,
    daily_vix,
    "Daily India VIX",
    required_columns=["india_vix"],
    on="date",
    validate="many_to_one",
)
master_dataset = left_merge_with_diagnostics(
    master_dataset,
    daily_risk_free_rate,
    "Daily Risk-Free Rate",
    required_columns=["risk_free_rate"],
    on="date",
    validate="many_to_one",
)

# Calendar-day time to expiry is retained for comparability with the original
# notebook, while trading-day time to expiry reflects actual observed sessions.
master_dataset["days_to_expiry"] = (
    master_dataset["expiry"] - master_dataset["date"]
).dt.days.astype("Int64")
master_dataset["time_to_expiry_days"] = master_dataset["days_to_expiry"]
master_dataset["trading_days_to_expiry"] = calculate_trading_days_to_expiry(
    master_dataset, trading_dates, start_col="date", expiry_col="expiry"
)
master_dataset["time_to_expiry_years"] = (
    master_dataset["trading_days_to_expiry"].astype("Float64") / 252.0
)

# IL&FS event variables are created here so later pricing, hedging, and event-study
# notebooks can use a consistent event definition without recalculating it.
master_dataset["event_date"] = EVENT_DATE
master_dataset["event_dummy"] = (master_dataset["date"] >= EVENT_DATE).astype("Int64")
master_dataset.loc[master_dataset["date"].isna(), "event_dummy"] = pd.NA
master_dataset["days_from_event"] = (master_dataset["date"] - EVENT_DATE).dt.days.astype("Int64")
master_dataset["weeks_from_event"] = np.floor(
    master_dataset["days_from_event"].astype("Float64") / 7
).astype("Int64")
master_dataset["pre_post_period"] = pd.Series(
    np.where(master_dataset["date"] < EVENT_DATE, "Pre", "Post"),
    index=master_dataset.index,
    dtype="string",
)
master_dataset.loc[master_dataset["date"].isna(), "pre_post_period"] = pd.NA

master_dataset["moneyness"] = np.where(
    master_dataset["strike"].gt(0),
    master_dataset["spot_price"] / master_dataset["strike"],
    np.nan,
)
master_dataset["year"] = master_dataset["date"].dt.year
master_dataset["month"] = master_dataset["date"].dt.month
master_dataset["week"] = master_dataset["date"].dt.isocalendar().week.astype("Int64")
master_dataset["day_of_week"] = master_dataset["date"].dt.day_name()
master_dataset["is_month_end"] = master_dataset["date"].dt.is_month_end
master_dataset = master_dataset.drop(columns=["__option_row_id"], errors="ignore")

preferred_column_order = [
    "date",
    "index",
    "expiry",
    "strike",
    "option_type",
    "spot_price",
    "future_price",
    "option_price",
    "option_open",
    "option_high",
    "option_low",
    "option_close",
    "option_settlement_price",
    "volume",
    "turnover",
    "open_interest",
    "change_in_oi",
    "india_vix",
    "risk_free_rate",
    "risk_free_rate_percent",
    "days_to_expiry",
    "time_to_expiry_days",
    "trading_days_to_expiry",
    "time_to_expiry_years",
    "event_date",
    "event_dummy",
    "days_from_event",
    "weeks_from_event",
    "pre_post_period",
    "moneyness",
    "daily_return_spot",
    "log_return_spot",
    "rolling_20d_volatility",
    "year",
    "month",
    "week",
    "day_of_week",
    "is_month_end",
    "spot_open",
    "spot_high",
    "spot_low",
    "shares_traded",
    "spot_turnover_inr_cr",
    "future_expiry",
    "future_match_type",
    "future_open",
    "future_high",
    "future_low",
    "future_close",
    "future_settlement_price",
    "future_days_to_expiry",
    "instrument",
]
remaining_columns = [col for col in master_dataset.columns if col not in preferred_column_order]
master_dataset = master_dataset[preferred_column_order + remaining_columns]
master_dataset = sort_and_reset(
    master_dataset, ["date", "index", "expiry", "strike", "option_type"]
)

if len(master_dataset) != len(options):
    raise ValueError(
        "The master dataset row count changed during merging. "
        "This would violate the option-level preservation requirement."
    )

if master_dataset[["spot_price", "future_price", "india_vix", "risk_free_rate"]].isna().any().any():
    missing_core = master_dataset[
        ["spot_price", "future_price", "india_vix", "risk_free_rate"]
    ].isna().sum()
    warnings_list.append(
        "Core merged fields still contain missing values: "
        + ", ".join(f"{col}={int(val)}" for col, val in missing_core.items() if val > 0)
    )

print(f"Master dataset dimensions: {master_dataset.shape}")


# %%
# ==================================================
# 10. DATA VALIDATION AND QUALITY CHECKS
# ==================================================

print_section("10. Data Validation and Quality Checks")

number_of_unique_trading_days = master_dataset["date"].nunique()
number_of_option_contracts = len(options)
number_of_futures_contracts = len(futures)
number_of_nifty_observations = int(master_dataset["index"].eq("NIFTY").sum())
number_of_banknifty_observations = int(master_dataset["index"].eq("BANKNIFTY").sum())
date_range = (master_dataset["date"].min(), master_dataset["date"].max())

duplicate_master_rows = int(master_dataset.duplicated().sum())
duplicate_option_keys = report_duplicate_keys(
    master_dataset, ["date", "index", "expiry", "strike", "option_type"]
)
merge_diagnostics_df = pd.DataFrame(merge_diagnostics)
conversion_failures_summary = summarise_conversion_failures()
trading_calendar_validation = create_trading_calendar_validation(
    spot_all, futures, options, india_vix, risk_free_rate
)
trading_calendar_validation_summary = (
    trading_calendar_validation.groupby("category", as_index=False)["issue_count"]
    .sum()
    .rename(columns={"issue_count": "date_count"})
    .sort_values("category")
)
contracts_by_instrument = (
    derivatives.groupby(["index", "instrument"], dropna=False)
    .size()
    .rename("contract_rows")
    .reset_index()
    .sort_values(["index", "instrument"])
)
contracts_by_expiry = (
    derivatives.groupby(["index", "instrument", "expiry"], dropna=False)
    .size()
    .rename("contract_rows")
    .reset_index()
    .sort_values(["index", "instrument", "expiry"])
)

print(f"Dataset dimensions: {master_dataset.shape}")
print(f"Number of unique trading days: {number_of_unique_trading_days:,}")
print(f"Number of option contracts: {number_of_option_contracts:,}")
print(f"Number of futures contracts: {number_of_futures_contracts:,}")
print(f"Number of NIFTY option observations: {number_of_nifty_observations:,}")
print(f"Number of BANKNIFTY option observations: {number_of_banknifty_observations:,}")
print(f"Date range: {date_range[0].date()} to {date_range[1].date()}")
print(f"Memory usage: {memory_usage_mb(master_dataset):.2f} MB")
print(f"Full-row duplicates in master dataset: {duplicate_master_rows:,}")
print(f"Duplicated option business keys: {len(duplicate_option_keys):,}")

print("\nMerge diagnostics summary:")
if merge_diagnostics_df.empty:
    print("No merge diagnostics were recorded.")
else:
    print(merge_diagnostics_df.to_string(index=False))

print("\nDuplicate option key summary:")
if duplicate_option_keys.empty:
    print("No duplicate option business keys detected.")
else:
    print(duplicate_option_keys.head(10).to_string(index=False))

print("\nMissing value table:")
master_missing_values = missing_value_table(master_dataset)
print(master_missing_values.to_string(index=False))

print("\nData types:")
print(master_dataset.dtypes)

print("\nDescriptive summary from pandas:")
try:
    descriptive_overview = master_dataset.describe(include="all", datetime_is_numeric=True).T
except TypeError:
    descriptive_overview = master_dataset.describe(include="all").T
print(descriptive_overview)

print("\nConversion failure summary:")
if conversion_failures_summary.empty:
    print("No date or numeric conversion failures detected in expected columns.")
else:
    print(conversion_failures_summary.to_string(index=False))

print("\nTrading calendar validation summary:")
if trading_calendar_validation_summary.empty:
    print("No trading-calendar discrepancies detected.")
else:
    print(trading_calendar_validation_summary.to_string(index=False))


# %%
# ==================================================
# 11. DESCRIPTIVE STATISTICS
# ==================================================

print_section("11. Descriptive Statistics")

summary_variables = [
    "spot_price",
    "future_price",
    "option_price",
    "option_close",
    "volume",
    "open_interest",
    "india_vix",
    "risk_free_rate",
    "risk_free_rate_percent",
    "days_to_expiry",
    "trading_days_to_expiry",
    "time_to_expiry_years",
    "moneyness",
    "event_dummy",
    "days_from_event",
    "weeks_from_event",
]

VARIABLE_DESCRIPTIONS = {
    "date": "Trading date for the option observation.",
    "index": "Underlying index, either NIFTY or BANKNIFTY.",
    "expiry": "Option contract expiry date.",
    "strike": "Option strike price.",
    "option_type": "Option type: CE for call options and PE for put options.",
    "spot_price": "Underlying spot closing price merged by Date and Index.",
    "future_price": "Matched futures settlement/close price used as the futures benchmark.",
    "future_match_type": "Whether the option used an expiry-matched future or the front-month fallback.",
    "option_price": "Option settlement price, falling back to close price when settlement is unavailable.",
    "volume": "Number of derivative contracts traded.",
    "turnover": "Derivative turnover reported in the source file.",
    "open_interest": "Open interest for the option contract.",
    "change_in_oi": "Change in open interest for the option contract.",
    "india_vix": "India VIX level aligned to the trading calendar.",
    "risk_free_rate": "Risk-free rate expressed as a decimal and aligned to the trading date.",
    "risk_free_rate_percent": "Risk-free rate expressed in percent before decimal conversion.",
    "days_to_expiry": "Calendar days from trading date to option expiry.",
    "time_to_expiry_days": "Calendar-day time to expiry retained for compatibility with the original notebook.",
    "trading_days_to_expiry": "Observed trading sessions from trading date to option expiry.",
    "time_to_expiry_years": "Trading-day time to expiry scaled by 252 sessions per year.",
    "event_date": "IL&FS default event date used consistently across the dissertation: 2018-09-04.",
    "event_dummy": "Indicator equal to 1 on/after the IL&FS event date and 0 before it.",
    "days_from_event": "Signed calendar days relative to 2018-09-04.",
    "weeks_from_event": "Signed integer weeks relative to 2018-09-04.",
    "pre_post_period": "Pre/Post label relative to the IL&FS event date.",
    "moneyness": "Spot price divided by option strike, included for later option-market analysis.",
    "daily_return_spot": "Daily simple return of the underlying spot index.",
    "log_return_spot": "Daily log return of the underlying spot index.",
    "rolling_20d_volatility": "Annualised 20-trading-day rolling realised spot volatility.",
    "future_expiry": "Matched futures expiry date.",
    "future_days_to_expiry": "Calendar days from trading date to matched futures expiry.",
}

summary_statistics = descriptive_statistics(master_dataset, summary_variables)
summary_statistics_by_index = grouped_descriptive_statistics(master_dataset, ["index"], summary_variables)
summary_statistics_by_option_type = grouped_descriptive_statistics(
    master_dataset, ["option_type"], summary_variables
)
summary_statistics_by_expiry = grouped_descriptive_statistics(master_dataset, ["expiry"], summary_variables)

# Appendix tables document the master dataset without altering any observations.
data_dictionary = create_data_dictionary(master_dataset, VARIABLE_DESCRIPTIONS)
outlier_report = create_outlier_report(master_dataset)
contract_summary = create_contract_summary(master_dataset, futures)

print(summary_statistics.to_string(index=False))

print("\nSummary statistics by index:")
print(summary_statistics_by_index.head(20).to_string(index=False))

print("\nOutlier report preview:")
print(outlier_report.head(20).to_string(index=False))

numeric_master = master_dataset.select_dtypes(include=[np.number]).copy()
correlation_matrix = numeric_master.corr()


# %%
# ==================================================
# 12. EXPLORATORY VISUALISATIONS
# ==================================================

print_section("12. Exploratory Visualisations")

# 12.1 Missing value heatmap.
heatmap_data = master_dataset.isna()
if len(heatmap_data) > 2000:
    heatmap_data = heatmap_data.sample(n=2000, random_state=42).sort_index()
    heatmap_title = "Missing Value Heatmap (2,000-Row Sample)"
else:
    heatmap_title = "Missing Value Heatmap"

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(heatmap_data, cbar=False, cmap=["#2A9D8F", "#E76F51"], ax=ax)
ax.set_title(heatmap_title)
ax.set_xlabel("Columns")
ax.set_ylabel("Observations")
save_figure(fig, "01_missing_value_heatmap.png")


# 12.2 Missing value bar chart.
fig, ax = plt.subplots(figsize=(12, 7))
missing_plot = master_missing_values.loc[master_missing_values["missing_count"].gt(0)].copy()
if missing_plot.empty:
    missing_plot = master_missing_values.head(1).copy()
sns.barplot(data=missing_plot, x="missing_count", y="column", color="#457B9D", ax=ax)
ax.set_title("Missing Values by Column")
ax.set_xlabel("Missing Count")
ax.set_ylabel("Column")
save_figure(fig, "02_missing_value_bar_chart.png")


# 12.3 Spot price time series.
fig, ax = plt.subplots(figsize=(13, 6))
sns.lineplot(data=spot_all, x="date", y="spot_price", hue="index", ax=ax, linewidth=1.8)
ax.set_title("Spot Price Time Series: NIFTY vs Bank NIFTY")
ax.set_xlabel("Date")
ax.set_ylabel("Spot Price")
ax.legend(title="Index")
save_figure(fig, "03_spot_price_time_series.png")


# 12.4 Futures price time series.
fig, ax = plt.subplots(figsize=(13, 6))
sns.lineplot(
    data=front_month_futures,
    x="date",
    y="future_price",
    hue="index",
    ax=ax,
    linewidth=1.8,
)
ax.set_title("Front-Month Futures Price Time Series")
ax.set_xlabel("Date")
ax.set_ylabel("Futures Price")
ax.legend(title="Index")
save_figure(fig, "04_futures_price_time_series.png")


# 12.5 India VIX time series.
fig, ax = plt.subplots(figsize=(13, 5.5))
sns.lineplot(data=daily_vix, x="date", y="india_vix", color="#6A4C93", ax=ax, linewidth=1.8)
ax.set_title("India VIX Time Series")
ax.set_xlabel("Date")
ax.set_ylabel("India VIX")
save_figure(fig, "05_india_vix_time_series.png")


# 12.6 Trading volume over time.
option_volume_daily = (
    master_dataset.groupby(["date", "index"], as_index=False)["volume"].sum()
)
fig, ax = plt.subplots(figsize=(13, 6))
sns.lineplot(
    data=option_volume_daily,
    x="date",
    y="volume",
    hue="index",
    ax=ax,
    linewidth=1.5,
)
ax.set_title("Option Trading Volume Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Total Option Contracts")
ax.legend(title="Index")
save_figure(fig, "06_trading_volume_over_time.png")


# 12.7 Open interest over time.
open_interest_daily = (
    master_dataset.groupby(["date", "index"], as_index=False)["open_interest"].sum()
)
fig, ax = plt.subplots(figsize=(13, 6))
sns.lineplot(
    data=open_interest_daily,
    x="date",
    y="open_interest",
    hue="index",
    ax=ax,
    linewidth=1.5,
)
ax.set_title("Option Open Interest Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Total Open Interest")
ax.legend(title="Index")
save_figure(fig, "07_open_interest_over_time.png")


# 12.8 Spot price distribution.
fig, ax = plt.subplots(figsize=(11, 6))
sns.histplot(data=spot_all, x="spot_price", hue="index", kde=True, bins=35, ax=ax)
ax.set_title("Distribution of Spot Prices")
ax.set_xlabel("Spot Price")
ax.set_ylabel("Frequency")
save_figure(fig, "08_spot_price_distribution.png")


# 12.9 Option price distribution.
fig, ax = plt.subplots(figsize=(11, 6))
sns.histplot(
    data=master_dataset,
    x="option_price",
    hue="index",
    kde=True,
    bins=50,
    ax=ax,
)
ax.set_title("Distribution of Option Prices")
ax.set_xlabel("Option Settlement Price")
ax.set_ylabel("Frequency")
save_figure(fig, "09_option_price_distribution.png")


# 12.10 Boxplots: spot, futures, options.
price_boxplot = pd.concat(
    [
        spot_all[["spot_price"]].rename(columns={"spot_price": "price"}).assign(price_type="Spot"),
        front_month_futures[["future_price"]]
        .rename(columns={"future_price": "price"})
        .assign(price_type="Futures"),
        master_dataset[["option_price"]]
        .rename(columns={"option_price": "price"})
        .assign(price_type="Options"),
    ],
    ignore_index=True,
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=price_boxplot, x="price_type", y="price", palette="Set2", ax=ax)
ax.set_title("Boxplots of Spot, Futures, and Option Prices")
ax.set_xlabel("Price Type")
ax.set_ylabel("Price")
save_figure(fig, "10_price_boxplots.png")


# 12.11 Correlation heatmap.
fig_width = max(12, min(24, 0.6 * len(correlation_matrix.columns)))
fig_height = max(10, min(22, 0.55 * len(correlation_matrix.columns)))
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
sns.heatmap(
    correlation_matrix,
    cmap="vlag",
    center=0,
    linewidths=0.25,
    linecolor="white",
    square=False,
    ax=ax,
)
ax.set_title("Correlation Heatmap of Numeric Variables")
ax.set_xlabel("Numeric Variables")
ax.set_ylabel("Numeric Variables")
save_figure(fig, "11_correlation_heatmap.png")


# 12.12 Daily returns.
fig, ax = plt.subplots(figsize=(13, 6))
sns.lineplot(
    data=spot_all,
    x="date",
    y="daily_return_spot",
    hue="index",
    ax=ax,
    linewidth=1.4,
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Daily Spot Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Return")
ax.legend(title="Index")
save_figure(fig, "12_daily_returns.png")


# 12.13 Rolling 20-day volatility.
fig, ax = plt.subplots(figsize=(13, 6))
sns.lineplot(
    data=spot_all,
    x="date",
    y="rolling_20d_volatility",
    hue="index",
    ax=ax,
    linewidth=1.6,
)
ax.set_title("Rolling 20-Day Realised Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualised Volatility")
ax.legend(title="Index")
save_figure(fig, "13_rolling_20d_volatility.png")


# 12.14 Option volume comparison.
fig, ax = plt.subplots(figsize=(9, 6))
volume_comparison = master_dataset.groupby("index", as_index=False)["volume"].sum()
sns.barplot(data=volume_comparison, x="index", y="volume", palette="Set2", ax=ax)
ax.set_title("Total Option Volume Comparison")
ax.set_xlabel("Index")
ax.set_ylabel("Total Option Contracts")
save_figure(fig, "14_option_volume_comparison.png")


# 12.15 Open interest comparison.
fig, ax = plt.subplots(figsize=(9, 6))
oi_comparison = master_dataset.groupby("index", as_index=False)["open_interest"].sum()
sns.barplot(data=oi_comparison, x="index", y="open_interest", palette="Set2", ax=ax)
ax.set_title("Total Open Interest Comparison")
ax.set_xlabel("Index")
ax.set_ylabel("Total Open Interest")
save_figure(fig, "15_open_interest_comparison.png")


# %%
# ==================================================
# 13. SAVE OUTPUTS
# ==================================================

print_section("13. Save Outputs")

master_output_path = OUTPUT_DIR / "master_dataset.csv"
summary_output_path = OUTPUT_DIR / "summary_statistics.csv"
missing_output_path = OUTPUT_DIR / "missing_values.csv"
correlation_output_path = OUTPUT_DIR / "correlation_matrix.csv"
summary_by_index_output_path = OUTPUT_DIR / "summary_statistics_by_index.csv"
summary_by_option_type_output_path = OUTPUT_DIR / "summary_statistics_by_option_type.csv"
summary_by_expiry_output_path = OUTPUT_DIR / "summary_statistics_by_expiry.csv"
data_dictionary_output_path = OUTPUT_DIR / "data_dictionary.csv"
outlier_report_output_path = OUTPUT_DIR / "outlier_report.csv"
contract_summary_output_path = OUTPUT_DIR / "contract_summary.csv"
trading_calendar_validation_output_path = OUTPUT_DIR / "trading_calendar_validation.csv"
merge_diagnostics_output_path = OUTPUT_DIR / "merge_diagnostics.csv"
conversion_failures_output_path = OUTPUT_DIR / "conversion_failures.csv"
contracts_by_instrument_output_path = OUTPUT_DIR / "contracts_by_instrument.csv"
contracts_by_expiry_output_path = OUTPUT_DIR / "contracts_by_expiry.csv"

master_output_path = save_csv_output(master_dataset, master_output_path, index=False)
summary_output_path = save_csv_output(summary_statistics, summary_output_path, index=False)
missing_output_path = save_csv_output(master_missing_values, missing_output_path, index=False)
correlation_output_path = save_csv_output(correlation_matrix, correlation_output_path, index=True)
summary_by_index_output_path = save_csv_output(
    summary_statistics_by_index, summary_by_index_output_path, index=False
)
summary_by_option_type_output_path = save_csv_output(
    summary_statistics_by_option_type, summary_by_option_type_output_path, index=False
)
summary_by_expiry_output_path = save_csv_output(
    summary_statistics_by_expiry, summary_by_expiry_output_path, index=False
)
data_dictionary_output_path = save_csv_output(data_dictionary, data_dictionary_output_path, index=False)
outlier_report_output_path = save_csv_output(outlier_report, outlier_report_output_path, index=False)
contract_summary_output_path = save_csv_output(contract_summary, contract_summary_output_path, index=False)
trading_calendar_validation_output_path = save_csv_output(
    trading_calendar_validation, trading_calendar_validation_output_path, index=False
)
merge_diagnostics_output_path = save_csv_output(
    merge_diagnostics_df, merge_diagnostics_output_path, index=False
)
conversion_failures_output_path = save_csv_output(
    conversion_failures_summary, conversion_failures_output_path, index=False
)
contracts_by_instrument_output_path = save_csv_output(
    contracts_by_instrument, contracts_by_instrument_output_path, index=False
)
contracts_by_expiry_output_path = save_csv_output(contracts_by_expiry, contracts_by_expiry_output_path, index=False)

created_output_paths = [
    master_output_path,
    summary_output_path,
    missing_output_path,
    correlation_output_path,
    summary_by_index_output_path,
    summary_by_option_type_output_path,
    summary_by_expiry_output_path,
    data_dictionary_output_path,
    outlier_report_output_path,
    contract_summary_output_path,
    trading_calendar_validation_output_path,
    merge_diagnostics_output_path,
    conversion_failures_output_path,
    contracts_by_instrument_output_path,
    contracts_by_expiry_output_path,
]

for output_path in created_output_paths:
    print(f"Saved output: {output_path}")


# %%
# ==================================================
# 14. DATA QUALITY REPORT
# ==================================================

print_section("14. Data Quality Report")

rows_before_cleaning = sum(item["rows_before"] for item in quality_audit.values())
rows_after_cleaning = (
    len(banknifty_spot)
    + len(nifty_spot)
    + len(india_vix)
    + len(risk_free_rate)
    + len(derivatives)
)
duplicates_removed = sum(item["duplicates_removed"] for item in quality_audit.values())
columns_removed = {
    label: details["columns_removed"]
    for label, details in quality_audit.items()
    if details["columns_removed"]
}
missing_values_remaining = int(master_dataset.isna().sum().sum())
execution_time_seconds = time.perf_counter() - EXECUTION_START_TIME
outlier_summary = pd.DataFrame(
    {
        "metric": [
            "variables_checked",
            "total_iqr_outliers",
            "total_z_score_outliers",
        ],
        "value": [
            len(outlier_report),
            int(outlier_report["IQR Outliers"].sum()) if not outlier_report.empty else 0,
            int(outlier_report["Z-score Outliers"].sum()) if not outlier_report.empty else 0,
        ],
    }
)

print("======================")
print("DATA QUALITY REPORT")
print("======================")
print(f"Rows before cleaning: {rows_before_cleaning:,}")
print(f"Rows after cleaning: {rows_after_cleaning:,}")
print(f"Duplicates removed: {duplicates_removed:,}")
print(f"Columns removed: {columns_removed if columns_removed else 'None'}")
print(f"Missing values remaining in master dataset: {missing_values_remaining:,}")
print(f"Date range: {date_range[0].date()} to {date_range[1].date()}")
print(f"Total option contracts: {number_of_option_contracts:,}")
print(f"Total futures contracts: {number_of_futures_contracts:,}")
print(f"Master dataset dimensions: {master_dataset.shape}")
print(f"Master dataset memory usage: {memory_usage_mb(master_dataset):.2f} MB")
print(f"Execution time: {execution_time_seconds:.2f} seconds")

print("\nMerge success statistics:")
if merge_diagnostics_df.empty:
    print("No merge diagnostics were recorded.")
else:
    print(merge_diagnostics_df.to_string(index=False))

print("\nDuplicate key summary:")
print(f"Full-row duplicates in master dataset: {duplicate_master_rows:,}")
if duplicate_option_keys.empty:
    print("No duplicate option business keys detected.")
else:
    print(duplicate_option_keys.head(10).to_string(index=False))

print("\nConversion failures:")
if conversion_failures_summary.empty:
    print("None")
else:
    print(conversion_failures_summary.to_string(index=False))

print("\nOutlier summary:")
print(outlier_summary.to_string(index=False))

print("\nTrading calendar validation summary:")
if trading_calendar_validation_summary.empty:
    print("No trading-calendar discrepancies detected.")
else:
    print(trading_calendar_validation_summary.to_string(index=False))

print("\nContracts by instrument:")
print(contracts_by_instrument.to_string(index=False))

print("\nContracts by expiry preview:")
print(contracts_by_expiry.head(20).to_string(index=False))
print(f"Contracts by expiry rows: {len(contracts_by_expiry):,}")

if warnings_list:
    print("\nWarnings:")
    for warning_message in warnings_list:
        print(f"- {warning_message}")
else:
    print("\nWarnings: None")

print("\nOutput files created:")
for output_path in created_output_paths:
    print(f"- {output_path}")
print(f"- Figures saved in: {FIGURES_DIR}")

print("\nStep 1 completed successfully. No pricing, Greeks, implied volatility, hedging, regression, or event study analysis was performed.")
