# Part C Regression Analysis Report

## Method
- Library: `statsmodels`
- Core API used for p-values: `statsmodels.api.OLS(...).fit()`
- Data split rule: rows 501-630 as test (130 rows), remaining rows as train.
- Preprocessing sets:
  - Set 1: standardized predictors using `StandardScaler` fit on train only.
  - Set 2: raw predictors (no transform).
  - Set 3: `log1p` transform on predictors.

## Data Check
- Total rows loaded: 1030
- Train rows: 900
- Test rows: 130

## Comparison Summary
- set1_standardized: significant (p<0.05) = 7, median p-value = 0.000921367, mean p-value = 0.0308143
- set2_raw: significant (p<0.05) = 7, median p-value = 0.000921367, mean p-value = 0.0308143
- set3_log1p: significant (p<0.05) = 5, median p-value = 4.04128e-06, mean p-value = 0.0986299

## Conclusion
- Best preprocessing method by this criterion is **set1_standardized** (more significant features first, then smaller median/mean p-values).
- Standardization usually preserves significance structure while improving numerical scale consistency.
- log1p may help when skewed predictors become more linearly related to target; final judgment is based on observed p-values above.

## Run Command
```bash
python part_c_regression_analysis.py
```