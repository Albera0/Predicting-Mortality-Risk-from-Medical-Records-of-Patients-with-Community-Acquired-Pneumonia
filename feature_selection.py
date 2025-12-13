import os
import pandas as pd
import xgboost as xgb


# Load data
df_clean = pd.read_csv("data\\NACef_FINAL_encoded.csv")

target = "gen_hosp_death"

# 只使用结局已知样本
df_model = df_clean[df_clean[target].notna()].copy()

X = df_model.drop(columns=["icu_death", "gen_hosp_death"])
y = df_model[target].astype(int)

# XGBoost for feature selection

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X, y)

# Feature importance

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

importance.to_csv("data/xgb_feature_importance.csv", index=False)


# Select top features

top_features = importance.head(40)["feature"].tolist()

df_selected = df_model[top_features + [target]]
df_selected.to_csv("data/NACef_selected_features.csv", index=False)

print("Feature selection completed")
print("Selected features:", len(top_features))
