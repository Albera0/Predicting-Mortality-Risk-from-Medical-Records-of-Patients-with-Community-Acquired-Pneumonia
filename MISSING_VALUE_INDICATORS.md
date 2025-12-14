# Missing Value Indicator Feature

## What is this code doing?

The code pattern:
```python
for col in FEATURES:
    df[col + "_missing"] = df[col].isna().astype(np.float32)
    df[col] = df[col].fillna(0)

FEATURES = FEATURES + [c + "_missing" for c in FEATURES]
```

This implements a **missing value indicator** pattern, which is a common preprocessing technique in machine learning:

### Purpose:
1. **Preserve information about missingness**: When we fill missing values (imputation), we lose information about which values were originally missing. This information can be valuable for predictive modeling.
2. **Create binary indicator features**: For each feature with potential missing values, create a new binary column that flags whether the original value was missing (1) or present (0).
3. **Expand the feature set**: Add these indicator columns to the feature list so they're included in model training.

### Example:
**Before:**
```
| age | bmi  |
|-----|------|
| 25  | 22.5 |
| 30  | NaN  |
| NaN | 28.1 |
```

**After:**
```
| age | bmi  | age_missing | bmi_missing |
|-----|------|-------------|-------------|
| 25  | 22.5 | 0           | 0           |
| 30  | 0    | 0           | 1           |
| 0   | 28.1 | 1           | 0           |
```

### Why is this useful?
- Missing values are often not random (Missing Not At Random - MNAR)
- The pattern of missingness itself can be predictive
- Example: Missing lab test results might indicate the test wasn't needed (patient was healthy) or wasn't performed (severe cases)

### Implementation in this project:
The code has been added to `preprocessing.py` at lines 302-307, where it:
1. Creates `_missing` indicator columns for all numeric features (`num_cols`)
2. Uses `np.float32` for memory efficiency
3. Then performs median imputation using `SimpleImputer`
4. These indicator columns are automatically included in the final dataset

This ensures that when the data is used for mortality prediction, the model can learn from both the imputed values AND the pattern of missingness.
