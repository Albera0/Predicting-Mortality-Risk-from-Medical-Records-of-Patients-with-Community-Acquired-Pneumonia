"""
Mission: Total Mortality prediction
Feature Prioritization + Encoding + Outlier Cleaning (FIXED)
"""

# icu_death and gen_hosp_death are not inputted as features,
# they are reserved as target for prediction


import pandas as pd
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("data\\NACef_Data.csv")
columns = df.columns.tolist()

print(f"original data: {df.shape[0]} rows, {df.shape[1]} columns")

# Delete highly missing data
missing_threshold = 0.95
missing_rates = df.isnull().sum() / len(df)
high_missing_cols = missing_rates[missing_rates > missing_threshold].index.tolist()

if high_missing_cols:
    print(f"\nfind {len(high_missing_cols)} high missing columns:")
    for col in high_missing_cols:
        miss_rate = missing_rates[col]
        print(f"  {col:30s}: {miss_rate * 100:5.1f}% missing")

    df = df.drop(columns=high_missing_cols)
    print(f"\n high missing columns has been deleted")


print(f"remaining: {df.shape[1]} columns")

# Set clinical priority
HIGH = [
    "sofa_72", "admission_sofa", "admission_curb", "admission_psi",
    "icu",
    "ventilation", "ventilation_type",
    "major_criteria", "sev_criteria", "criteria",
    "res_pattern", "res_pattern_2",
    "int_24h"
]

HIGH_IMPORTANT = [
    "isolated_micro", "coinfection", "coinfection_microorg",
    "etio_pneumo", "etio_pneumo_patogen",
    "failure_cause",
    "secund_infec", "non_pulm_infec",
    "cv_comp", "cv",
    "age", "comorbid", "copd", "gold",
    "ab_24h", "ab_empiric", "ab_empiric_2",
    "minor_criteria", "main_diagnosis",
    "2_rt_pcr_covid", "sosp_covid",
    "geriatric_home", "bedridden",
]

MEDIUM = [
    "bmi",
    "ab_prev", "ab_12month", "ab_conjug",
    "prev_ab_num", "prev_ab_days",
    "prev_infec", "num_prev_infec",
    "urg_12_month", "num_urg",
    "days_ab", "ant_mdrd",
    "hosp_stay", "gender"
]

LOW = [
    "ant_taba",
    "vac_influenza", "vac_covid",
]

DROP= [

    # ID / Study Meta
    "record_id",
    "which_study", "another_study",

    # Vaccination details
    "vac_neumo_type",
    "vac_neumococo",

    # Pure date columns
    "date_vac_influenza", "date_vac_neumo_2",
    "date_infec", "date_hosp", "prev_ab_date",
    "date_extub", "date_traqueost",
    "dicharge_date", "date_non_pulmonar",
    "admission_date",

    # Low-importance demographic/household
    "health_work", "rural_work",
    "living_space",

    # Potential redundancy with BMI
    "weight", "height",

    # Outcome / future info in case of leakage
    "icu_discharge",
    "live_discharge",
    "extub",
    "traqueost",
    "clinical_resp",
    "treatm_fail_cause"
]


def score_feature(col):
    # Assign clinical priority score to each feature
    col_lower = col.lower()
    if any(k in col_lower for k in DROP):
        return 1
    if any(k in col_lower for k in HIGH):
        return 10
    if any(k in col_lower for k in HIGH_IMPORTANT):
        return 9
    if any(k in col_lower for k in MEDIUM):
        return 7
    if any(k in col_lower for k in LOW):
        return 4
    return 3



# Drop predefined columns
df_clean = df.copy()

cols_to_drop = [col for col in DROP if col in df_clean.columns]
if cols_to_drop:
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"Remaining columns after Dropping: {df_clean.shape[1]} 列")

clinical_score = {col: score_feature(col) for col in df_clean.columns}
pd.DataFrame({
    "feature": df_clean.columns,
    "clinical_score": [clinical_score[c] for c in df_clean.columns]
}).to_csv("data/clinical_prior_score_auto.csv", index=False)


# Inspect remaining columns
for col in df_clean.columns:
    print(col)
print("total columns:", len(df_clean.columns))

# Define numeric / categorical vars

continuous_numeric = [
    "age", "bmi", "prev_ab_num", "prev_ab_days",
    "num_prev_infec", "num_urg",
    "days_ab", "hosp_stay",
    "admission_sofa", "admission_curb", "admission_psi",
    "sofa_72",
]


count_vars = [
    "num_prev_infec",
    "num_urg",
    "prev_ab_num",
    "prev_ab_days",
]

ordinal_vars = [
    "gold",
    "ventilation", "ventilation_type",
]

categorical_vars = [
    "ab_24h", "ab_prev",
    "minor_criteria", "major_criteria", "criteria", "sev_criteria",
    "int_24h",
    "isolated_micro", "coinfection_microorg",
    "etio_pneumo", "etio_pneumo_patogen",
    "failure_cause",
    "main_diagnosis", "cv",
    "ab_empiric", "ab_empiric_2", "ab_conjug", "ab_conjug2",
]

binary_vars = [
    "gender",
    "geriatric_home", "bedridden",
    "vac_influenza", "vac_covid",
    "sosp_covid",
    "coinfection", "secund_infec",
    "icu",
    "prev_infec", "urg_12_month", "ab_12month",
    "ant_taba",
]

outcome_vars = ["icu_death", "gen_hosp_death"]

'''
multi label one-hot coding-ab_24h
'''
def multi_hot_encode(
    df,
    col,
    sep=",",
    add_none_col=True,
    drop_original=True,
):

    if col not in df.columns:
        print(f"skip {col} not existing columns")
        return df

    # Standardize String Format
    s = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace("-", "/", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.replace("None", "", regex=False)
        .str.replace("none", "", regex=False)
        .str.replace("nan", "", regex=False)
    )

    def split_to_list(x):
        if x == "":
            return []
        return [item.strip() for item in x.split(sep) if item.strip() != ""]

    list_col = f"{col}__list"
    df[list_col] = s.apply(split_to_list)

    # Collect labels
    from itertools import chain
    all_labels = sorted(set(chain.from_iterable(df[list_col])))

    print(f"{col}: {len(all_labels)} labels")

    # multi-hot
    for label in all_labels:
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("<", "")
            .replace(">", "")
        )
        new_col = f"{col}_{safe_label}"
        df[new_col] = df[list_col].apply(lambda lst: int(label in lst))

    # None
    if add_none_col:
        df[f"{col}_None"] = df[list_col].apply(lambda lst: int(len(lst) == 0))

    # clearing
    if drop_original:
        df = df.drop(columns=[col, list_col])
    else:
        df = df.drop(columns=[list_col])

    return df


# multi-hot encoding for following features
for col in ["ab_24h", "comorbid", "minor_criteria", "cv", "ab_prev", "int_24h", "secund_infec"]:
    df_clean = multi_hot_encode(df_clean, col)


# age over 90 encoded as 95
if "age" in df_clean.columns:
    df_clean["age"] = (
        df_clean["age"]
        .astype(str)
        .str.strip()
        .replace({">90": "95"})
    )

    df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce")

def encode_major_criteria_binary(df):

    if "major_criteria" not in df.columns:
        return df

    df["major_criteria_flag"] = (
        df["major_criteria"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: 0 if x in ["", "nan", "none", "0"] else 1)
    )

    df = df.drop(columns=["major_criteria"])
    return df


df_clean = encode_major_criteria_binary(df_clean)


# Only keep the consecutive columns of values that actually exist in the current DataFrame
num_cols = [c for c in continuous_numeric if c in df_clean.columns]

print("Columns to be filled with values：", num_cols)

imputer = SimpleImputer(strategy="median")
df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])

df_clean.to_csv("data/NACef_FINAL_encoded.csv", index=False)
print("\nfinal form saved：NACef_FINAL_encoded.csv")
print("final dimension:", df_clean.shape)




