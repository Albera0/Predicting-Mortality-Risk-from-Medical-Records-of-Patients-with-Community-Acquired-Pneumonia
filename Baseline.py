import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Database/NACef_selected_features.csv")
TARGET = "gen_hosp_death"
FEATURES = [c for c in df.columns if c != TARGET]

# Missing indicator
for col in FEATURES:
    df[col + "_missing"] = df[col].isna().astype(np.float32)
    df[col] = df[col].fillna(0)
FEATURES = FEATURES + [c + "_missing" for c in FEATURES]

X = df[FEATURES].values
y = df[TARGET].values

print("Input shape:", X.shape)
print("Positive rate:", y.mean())

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auroc_scores = []
auprc_scores = []
all_precisions = []
all_recalls = []

# Cross-validation
fold = 1
for train_idx, test_idx in skf.split(X, y):
    print(f"\n===== Fold {fold} =====")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Class weight
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    cw = {0: weights[0], 1: weights[1]}

    # Logistic Regression with L2
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight=cw,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred = model.predict_proba(X_test)[:, 1]

    # Metrics
    auroc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)

    print(f"Fold {fold} AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")

    auroc_scores.append(auroc)
    auprc_scores.append(auprc)

    # Interpolate PR curve
    recall_grid = np.linspace(0, 1, 100)
    interp_prec = np.interp(recall_grid, recall[::-1], precision[::-1])
    all_precisions.append(interp_prec)
    all_recalls.append(recall_grid)

    fold += 1

# Summary
print("\n=== Logistic Regression CV Summary ===")
print(f"Mean AUROC: {np.mean(auroc_scores):.3f} ± {np.std(auroc_scores):.3f}")
print(f"Mean AUPRC: {np.mean(auprc_scores):.3f} ± {np.std(auprc_scores):.3f}")


# Plot mean PR curve
mean_prec = np.mean(all_precisions, axis=0)
std_prec = np.std(all_precisions, axis=0)
recall_grid = all_recalls[0]

plt.figure(figsize=(8,6))
plt.plot(recall_grid, mean_prec, label="Logistic Regression", color="darkorange")
plt.fill_between(
    recall_grid,
    mean_prec - std_prec,
    mean_prec + std_prec,
    alpha=0.2,
    color="darkorange"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Logistic Regression Baseline)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logistic_pr_curve.png", dpi=300)
plt.show()
