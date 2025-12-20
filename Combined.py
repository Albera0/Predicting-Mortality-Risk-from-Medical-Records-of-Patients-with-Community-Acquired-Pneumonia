import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Multiply, Add, BatchNormalization, Lambda
from tensorflow.keras import backend as K

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load and preprocess data
DATA_PATH = "Database/NACef_selected_features.csv"
TARGET = "gen_hosp_death"

continuous_vars = [
    "age","hosp_stay","days_ab","admission_sofa","sofa_72",
    "admission_curb","admission_psi","gold","isolated_micro",
    "coinfection_microorg","res_pattern","ab_empiric_2"
]

df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != TARGET]
df = df[features + [TARGET]].copy()

binary_vars = [c for c in df.columns if c not in continuous_vars + [TARGET]]

for col in continuous_vars:
    df[col + "_missing"] = df[col].isna().astype(np.float32)

df[continuous_vars + binary_vars] = df[continuous_vars + binary_vars].fillna(0)

FEATURES = continuous_vars + binary_vars + [c + "_missing" for c in continuous_vars]

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Sparsemax activation
class Sparsemax(tf.keras.layers.Layer):
    def call(self, z):
        z_sorted = tf.sort(z, direction="DESCENDING", axis=-1)
        z_cumsum = tf.cumsum(z_sorted, axis=-1)
        k = tf.range(1, tf.shape(z)[-1] + 1, dtype=z.dtype)
        k = tf.reshape(k, (1, -1))
        support = tf.cast(1 + k * z_sorted > z_cumsum, z.dtype)
        k_z = tf.reduce_sum(support, axis=-1, keepdims=True)
        tau = (tf.reduce_sum(support * z_sorted, axis=-1, keepdims=True) - 1) / k_z
        return tf.maximum(z - tau, 0.)
    
# GLU Block
def GLU(x, units, name):
    linear = Dense(units, activation=None, name=name+"_linear")(x)
    gate = Dense(units, activation="sigmoid", name=name+"_gate")(x)
    return Multiply()([linear, gate])

# Feature Transformer（shared + specific）
def FeatureTransformer(x, hidden_dim, name, dropout=0.2):
    h = Dense(hidden_dim*2, activation=None, name=name+"_dense1")(x)
    h = BatchNormalization()(h)
    h = GLU(h, hidden_dim, name+"_glu1")
    h = Dense(hidden_dim*2, activation=None, name=name+"_dense2")(h)
    h = BatchNormalization()(h)
    h = GLU(h, hidden_dim, name+"_glu2")
    h = Dropout(dropout)(h)
    return h

# Attentive Transformer
def AttentiveTransformer(x_input, input_dim, prior, name):
    mask_logits = Dense(input_dim, name=name+"_fc")(x_input)
    mask = Sparsemax()(mask_logits * prior)
    new_prior = Lambda(lambda p_m: p_m[0]*(1.5 - p_m[1]), name=name+"_prior")([prior, mask])
    masked_features = Multiply()([x_input, mask])
    return masked_features, new_prior, mask
    
# TabNet-like Model
def build_tabnet(input_dim, hidden_dim=64, n_steps=3, dropout=0.1):
    inputs = Input(shape=(input_dim,))
    prior = Lambda(lambda x: tf.ones_like(x))(inputs)
    decision_outputs = []
    masks_all = []

    for step in range(n_steps):
        x_masked, prior, mask = AttentiveTransformer(inputs, input_dim, prior, name=f"attn_{step}")
        masks_all.append(mask)
        h = FeatureTransformer(x_masked, hidden_dim, name=f"ft_{step}", dropout=dropout)
        decision_outputs.append(h)

    out = Add()(decision_outputs)
    outputs = Dense(1, activation="sigmoid")(out)

    model = Model(inputs, outputs)
    return model, masks_all


# Train
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}

tabnet_model, tabnet_masks = build_tabnet(X_train.shape[1])
tabnet_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auroc")]
)
tabnet_model.fit(X_train, y_train, epochs=50, batch_size=32, class_weight=class_weights, verbose=1)


# Evaluate
y_pred_tabnet = tabnet_model.predict(X_test).ravel()
fpr_tabnet, tpr_tabnet, _ = roc_curve(y_test, y_pred_tabnet)
roc_auc_tabnet = auc(fpr_tabnet, tpr_tabnet)
precision, recall, _ = precision_recall_curve(y_test, y_pred_tabnet)
auprc_tabnet = auc(recall, precision)
print(f"TabNet AUROC: {roc_auc_tabnet:.3f}")
print(f"TabNet AUPRC: {auprc_tabnet:.3f}")


# Logistic Regression baseline
logit_model = LogisticRegression(
    penalty="l2", C=1.0, solver="liblinear",
    class_weight=class_weights, max_iter=1000
)
logit_model.fit(X_train, y_train)
y_pred_logit = logit_model.predict_proba(X_test)[:, 1]

fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_logit)
roc_auc_logit = auc(fpr_logit, tpr_logit)

# ROC Curves
plt.figure(figsize=(8,6))
plt.plot(fpr_logit, tpr_logit, label=f"Logistic Regression (AUROC={roc_auc_logit:.3f})")
plt.plot(fpr_tabnet, tpr_tabnet, label=f"TabNet (AUROC={roc_auc_tabnet:.3f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# TabNet feature importance
mask_model = Model(inputs=tabnet_model.input, outputs=tabnet_masks)
masks_values = mask_model.predict(X_test, batch_size=32)
importances = np.mean(np.stack(masks_values, axis=0), axis=(0,1))

# Without missing indicators
original_feature_indices = [i for i, f in enumerate(FEATURES) if not f.endswith("_missing")]
importances_orig = importances[original_feature_indices]
features_orig = [FEATURES[i] for i in original_feature_indices]

# Top 20 features
top_idx = np.argsort(importances_orig)[-20:][::-1]
top_features = [features_orig[i] for i in top_idx]
top_values = importances_orig[top_idx]

plt.figure(figsize=(10,6))
plt.barh(range(len(top_values))[::-1], top_values, color="skyblue")
plt.yticks(range(len(top_features)), top_features[::-1])
plt.xlabel("Average Mask Value (Importance)")
plt.title("Top 20 Feature Importance (TabNet, excluding missing)")
plt.tight_layout()
plt.savefig("tabnet_feature_importance.png", dpi=300)
plt.show()

# Save results
# plt.savefig("tabnet_feature_importance.png", dpi=300)
# plt.show()