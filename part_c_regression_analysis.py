from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


EXPECTED_COLUMNS = 9
TEST_START_1BASED = 501
TEST_END_1BASED = 630
SIGNIFICANCE_LEVEL = 0.05


def clean_column_name(name: str) -> str:
    cleaned = re.sub(r"\(.*?\)", "", str(name))
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    cleaned = cleaned.replace("%", "pct")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def canonical_feature_names() -> List[str]:
    return [
        "Cement",
        "Slag",
        "Fly_Ash",
        "Water",
        "Superplasticizer",
        "Coarse_Aggregate",
        "Fine_Aggregate",
        "Age",
    ]


def ensure_data_file(project_dir: Path) -> Path:
    data_path = project_dir / "Concrete_Data.csv"
    if data_path.exists():
        return data_path

    fallback_urls = [
        "https://raw.githubusercontent.com/selva86/datasets/master/Concrete_Data.csv",
        "https://raw.githubusercontent.com/ageron/data/main/concrete/Concrete_Data.csv",
    ]

    for url in fallback_urls:
        try:
            df_url = pd.read_csv(url)
            if df_url.shape[1] == EXPECTED_COLUMNS:
                df_url.to_csv(data_path, index=False)
                print(f"[INFO] Concrete_Data.csv not found locally, downloaded from: {url}")
                return data_path
        except Exception:
            continue

    raise FileNotFoundError(
        "Concrete_Data.csv not found in project directory and automatic download failed. "
        "Please place Concrete_Data.csv in the same folder as this script."
    )


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file {data_path.name}: {exc}") from exc

    if df.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns (8 predictors + 1 target), "
            f"but got {df.shape[1]} columns."
        )

    cleaned_cols = [clean_column_name(col) for col in df.columns]
    df.columns = cleaned_cols

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            "Non-numeric or missing values detected after numeric conversion in columns: "
            + ", ".join(na_cols)
        )

    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    test_start_idx = TEST_START_1BASED - 1
    test_end_exclusive = TEST_END_1BASED

    if n_rows < TEST_END_1BASED:
        raise ValueError(
            f"Dataset has {n_rows} rows, but requires at least {TEST_END_1BASED} rows "
            "to apply the required split (rows 501-630 as test)."
        )

    test_df = df.iloc[test_start_idx:test_end_exclusive].copy()
    train_df = pd.concat([df.iloc[:test_start_idx], df.iloc[test_end_exclusive:]], axis=0).copy()

    return train_df, test_df


def fit_ols_with_pvalues(
    X_train_proc: pd.DataFrame,
    y_train: pd.Series,
    set_name: str,
) -> pd.DataFrame:
    X_train_const = sm.add_constant(X_train_proc, has_constant="add")
    model = sm.OLS(y_train, X_train_const).fit()

    rows = []
    for feature in X_train_proc.columns:
        coef = float(model.params[feature])
        p_val = float(model.pvalues[feature])
        rows.append(
            {
                "set_name": set_name,
                "feature": feature,
                "coefficient": coef,
                "p_value": p_val,
                "significant_0_05": bool(p_val < 0.05),
                "significant_0_01": bool(p_val < 0.01),
            }
        )

    return pd.DataFrame(rows)


def prepare_sets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    sets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    scaler = StandardScaler()
    X_train_std = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_std = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    sets["set1_standardized"] = (X_train_std, X_test_std)

    sets["set2_raw"] = (X_train.copy(), X_test.copy())

    if (X_train < 0).any().any() or (X_test < 0).any().any():
        raise ValueError(
            "Negative values found in predictors; cannot apply log1p safely for Set 3."
        )
    sets["set3_log1p"] = (np.log1p(X_train), np.log1p(X_test))

    return sets


def build_summary_table(detailed_df: pd.DataFrame) -> pd.DataFrame:
    pivot = detailed_df.pivot(index="feature", columns="set_name", values="p_value").reset_index()
    rename_map = {
        "set1_standardized": "p_value_set1_standardized",
        "set2_raw": "p_value_set2_raw",
        "set3_log1p": "p_value_set3_log1p",
    }
    pivot = pivot.rename(columns=rename_map)

    set_cols = [
        "p_value_set1_standardized",
        "p_value_set2_raw",
        "p_value_set3_log1p",
    ]
    pivot["best_set_by_min_p"] = (
        pivot[set_cols]
        .idxmin(axis=1)
        .map(
            {
                "p_value_set1_standardized": "set1_standardized",
                "p_value_set2_raw": "set2_raw",
                "p_value_set3_log1p": "set3_log1p",
            }
        )
    )
    return pivot


def summarize_sets(detailed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = detailed_df.groupby("set_name", as_index=False).agg(
        significant_count_0_05=("significant_0_05", "sum"),
        median_p_value=("p_value", "median"),
        mean_p_value=("p_value", "mean"),
    )
    return grouped


def pick_best_method(metrics_df: pd.DataFrame) -> str:
    ranked = metrics_df.sort_values(
        by=["significant_count_0_05", "median_p_value", "mean_p_value"],
        ascending=[False, True, True],
    )
    return str(ranked.iloc[0]["set_name"])


def write_report(
    report_path: Path,
    data_rows: int,
    train_rows: int,
    test_rows: int,
    metrics_df: pd.DataFrame,
    best_method: str,
) -> None:
    lines = [
        "# Part C Regression Analysis Report",
        "",
        "## Method",
        "- Library: `statsmodels`",
        "- Core API used for p-values: `statsmodels.api.OLS(...).fit()`",
        "- Data split rule: rows 501-630 as test (130 rows), remaining rows as train.",
        "- Preprocessing sets:",
        "  - Set 1: standardized predictors using `StandardScaler` fit on train only.",
        "  - Set 2: raw predictors (no transform).",
        "  - Set 3: `log1p` transform on predictors.",
        "",
        "## Data Check",
        f"- Total rows loaded: {data_rows}",
        f"- Train rows: {train_rows}",
        f"- Test rows: {test_rows}",
        "",
        "## Comparison Summary",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['set_name']}: "
            f"significant (p<0.05) = {int(row['significant_count_0_05'])}, "
            f"median p-value = {row['median_p_value']:.6g}, "
            f"mean p-value = {row['mean_p_value']:.6g}"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            f"- Best preprocessing method by this criterion is **{best_method}** "
            "(more significant features first, then smaller median/mean p-values).",
            "- Standardization usually preserves significance structure while improving numerical scale consistency.",
            "- log1p may help when skewed predictors become more linearly related to target; final judgment is based on observed p-values above.",
            "",
            "## Run Command",
            "```bash",
            "python part_c_regression_analysis.py",
            "```",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    summary_path = project_dir / "part_c_pvalues_summary.csv"
    detailed_path = project_dir / "part_c_coeffs_pvalues_detailed.csv"
    report_path = project_dir / "part_c_report.md"

    try:
        data_path = ensure_data_file(project_dir)
        df = load_and_validate_data(data_path)

        if len(df) != 1030:
            print(
                f"[WARN] Loaded {len(df)} rows (expected typical dataset size: 1030). "
                "Continuing with required split rule."
            )

        feature_cols = canonical_feature_names()
        target_col = "Concrete_Compressive_Strength"

        predictors = df.iloc[:, :8].copy()
        target = df.iloc[:, -1].copy()

        predictors.columns = feature_cols
        target.name = target_col

        split_df = predictors.copy()
        split_df[target_col] = target
        train_df, test_df = split_train_test(split_df)

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]

        preprocessed_sets = prepare_sets(X_train, X_test)

        all_detailed = []
        for set_name, (X_train_proc, _X_test_proc) in preprocessed_sets.items():
            detailed = fit_ols_with_pvalues(X_train_proc, y_train, set_name)
            all_detailed.append(detailed)

        detailed_df = pd.concat(all_detailed, ignore_index=True)
        summary_df = build_summary_table(detailed_df)

        detailed_df.to_csv(detailed_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        metrics_df = summarize_sets(detailed_df)
        best_method = pick_best_method(metrics_df)

        write_report(
            report_path=report_path,
            data_rows=len(df),
            train_rows=len(train_df),
            test_rows=len(test_df),
            metrics_df=metrics_df,
            best_method=best_method,
        )

        print("\n=== Part C Regression Analysis Completed ===")
        print(f"Data file: {data_path.name}")
        print(f"Total rows loaded: {len(df)}")
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        print("Three OLS models fitted: set1_standardized, set2_raw, set3_log1p")
        print("\nGenerated files:")
        print(f"- {summary_path.name}")
        print(f"- {detailed_path.name}")
        print(f"- {report_path.name}")

        print("\nSignificant feature counts (p < 0.05):")
        for _, row in metrics_df.iterrows():
            print(f"- {row['set_name']}: {int(row['significant_count_0_05'])}")

        print(f"\nBest preprocessing method: {best_method}")
        print("\nReport summary:")
        print("- Uses statsmodels.api.OLS(...).fit() to obtain p-values.")
        print("- Compares significant feature counts and p-value central tendency across 3 sets.")

        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
