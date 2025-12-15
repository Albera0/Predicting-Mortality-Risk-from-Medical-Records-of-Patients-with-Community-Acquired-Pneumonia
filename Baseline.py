import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Load data
# Load the data
DATA_PATH = "Database/NACef_selected_features.csv"
Target = "gen_hosp_death"

continuous_vars = [
    "age",
    "hosp_stay",
    "days_ab",
    "admission_sofa",
    "sofa_72",
    "admission_curb",
    "admission_psi",
    "gold",
    "isolated_micro",
    "coinfection_microorg",
    "res_pattern",
    "ab_empiric_2"
]

df = pd.read_csv(DATA_PATH)
Features = [c for c in df.columns if c != Target]
df = df[Features + [Target]].copy()

binary_vars = [
    c for c in df.columns
    if c not in continuous_vars + [Target]
]


# Missing values handling
for col in continuous_vars:
    df[col + "_missing"] = df[col].isna().astype(np.float32)

df[continuous_vars + binary_vars] = df[continuous_vars + binary_vars].fillna(0)


Features = (
    continuous_vars
    + binary_vars
    + [c + "_missing" for c in continuous_vars]
)

X = df[Features].values
y = df[Target].values

print("Input shape:", X.shape)
print("Positive rate:", y.mean())

# Single stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Class weights
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
cw = {0: weights[0], 1: weights[1]}

# Logistic Regression
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    class_weight=cw,
    max_iter=1000
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict_proba(X_test)[:, 1]

# Metrics
auroc = roc_auc_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auprc = auc(recall, precision)

print("\n=== Logistic Regression Baseline ===")
print(f"AUROC: {auroc:.3f}")
print(f"AUPRC: {auprc:.3f}")

# Plot PR curve
plt.figure(figsize=(8,6))
plt.plot(recall, precision, label="Logistic Regression", color="darkorange")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Logistic Regression Baseline)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logistic_pr_curve.png", dpi=300)
plt.show()
