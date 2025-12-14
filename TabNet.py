import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Split and class weights
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


# TensorFlow/Keras training
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Multiply, Add
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score



# Load the data
DATA_PATH = "Database/NACef_selected_features.csv"
Targe = "gen_hosp_death"

df = pd.read_csv(DATA_PATH)
Features = [c for c in df.columns if c != Targe]
df = df[Features + [Targe]].copy()

# Missing values handling
for col in Features:
    df[col + "_missing"] = df[col].isna().astype(np.float32)
    df[col] = df[col].fillna(0)

Features = Features + [c + "_missing" for c in Features]

# Train-test split
X = df[Features].values.astype(np.float32)
y = df[Targe].values.astype(np.float32)

print("Input shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])


# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

auprc_scores = []
auroc_scores = []

all_precisions = []
all_recalls = []

# Build the TablNet model
def tabnet_like(input_dim, hidden_dim=32, n_steps=2, dropout=0.1, l2_lambda=0.001):
    inputs = Input(shape=(input_dim,))
    h = Dense(hidden_dim, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(inputs)
    decisions_list = []

    for i in range(n_steps):
        # Step mask
        mask = Dense(input_dim, activation="softmax", name=f"mask_{i}")(h)
        masked = Multiply(name=f"masked_{i}")([inputs, mask])

        # Decision step
        decision = Dense(hidden_dim, activation="relu", name=f"decision_{i}")(masked)
        decision = Dropout(dropout, name=f"dropout_{i}")(decision)
        decisions_list.append(decision)
        h = decision

    # Aggregate decisions
    if n_steps > 1:
        aggregated = Add(name="agg_all")(decisions_list)
    else:
        aggregated = decisions_list[0]

    outputs = Dense(1, activation="sigmoid")(aggregated)
    model = Model(inputs, outputs, name="Simplified_TabNet")
    return model

# Cross-validation training
fold = 1
plt.figure(figsize=(8,6))

for train_idx, test_idx in skf.split(X, y):
    print(f"\n===== Fold {fold} =====")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # class weight
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {0: weights[0], 1: weights[1]}

    model = tabnet_like(X_train.shape[1], hidden_dim=32, n_steps=2, dropout=0.1, l2_lambda=0.001)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[AUC(name="auroc"), AUC(curve="PR", name="auprc"),
            Precision(name="precision"), Recall(name="recall")]
    )
    #model.summary()


    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, class_weight=class_weights, verbose=0)

    y_pred = model.predict(X_test).ravel()

    # Evaluate the model
    auroc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)
    print(f"Fold {fold} AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")

    # Store scores
    auroc_scores.append(auroc)
    auprc_scores.append(auprc)

    all_precisions.append(np.interp(np.linspace(0,1,100), recall[::-1], precision[::-1]))
    all_recalls.append(np.linspace(0,1,100))
    
    fold += 1

print("\n=== Cross-Validation Summary ===")
print(f"Mean AUROC: {np.mean(auroc_scores):.3f} ± {np.std(auroc_scores):.3f}")
print(f"Mean AUPRC: {np.mean(auprc_scores):.3f} ± {np.std(auprc_scores):.3f}")

# Plot curve
mean_prec = np.mean(all_precisions, axis=0)
std_prec = np.std(all_precisions, axis=0)
recall_grid = all_recalls[0]

plt.figure(figsize=(8,6))
plt.plot(recall_grid, mean_prec, color="blue", label="Mean PR")
plt.fill_between(recall_grid, mean_prec - std_prec, mean_prec + std_prec, color="blue", alpha=0.2, label="±1 SD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Cross-Validated Precision-Recall Curve (TabNet-style)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tabnet_pr_curve.png", dpi=300)
plt.show()