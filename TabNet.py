import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Multiply, Add
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

K.clear_session()
gc.collect()


# Load and preprocess data
DATA_PATH = "Database/NACef_selected_features.csv"
TARGET = "gen_hosp_death"

continuous_vars = [
    "age","hosp_stay","days_ab","admission_sofa","sofa_72",
    "admission_curb","admission_psi","gold","isolated_micro",
    "coinfection_microorg","res_pattern","ab_empiric_2"
]

# Load CSV
df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != TARGET]
df = df[features + [TARGET]].copy()

# Identify binary features
binary_vars = [c for c in df.columns if c not in continuous_vars + [TARGET]]

# Handle missing values
for col in continuous_vars:
    df[col + "_missing"] = df[col].isna().astype(np.float32)
df[continuous_vars + binary_vars] = df[continuous_vars + binary_vars].fillna(0)

# Final features
features = continuous_vars + binary_vars + [c + "_missing" for c in continuous_vars]

# Train-test split
X = df[features].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)
print("Input shape:", X.shape)


# Sparsemax activation
class Sparsemax(tf.keras.layers.Layer):
    def call(self, inputs):
        z = inputs
        z_sorted = tf.sort(z, direction="DESCENDING", axis=-1)
        z_cumsum = tf.cumsum(z_sorted, axis=-1)

        k = tf.range(1, tf.shape(z)[-1] + 1, dtype=z.dtype)
        k = tf.reshape(k, (1, -1))

        support = tf.cast(1 + k * z_sorted > z_cumsum, z.dtype)
        k_z = tf.reduce_sum(support, axis=-1, keepdims=True)

        tau = (tf.reduce_sum(support * z_sorted, axis=-1, keepdims=True) - 1) / k_z
        return tf.maximum(z - tau, 0.)


# GLU block
def glu_block(x, units, name):
    linear = Dense(units, activation=None, name=f"{name}_linear")(x)
    gate = Dense(units, activation="sigmoid", name=f"{name}_gate")(x)
    return Multiply(name=f"{name}_glu")([linear, gate])


# Attentive Transformer
class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, input_dim, l2_lambda=0.001, gamma=1.5, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.gamma = gamma
        self.dense = Dense(input_dim, kernel_regularizer=regularizers.l2(l2_lambda))
        self.sparsemax = Sparsemax()
        # Lambda layer to safely compute mean on KerasTensor
        self.mean_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims=True))

    def call(self, inputs, prior):
        mask_logits = self.dense(inputs)
        mean_prior = self.mean_layer(prior)
        mask = self.sparsemax(mask_logits * mean_prior)
        new_prior = prior * (self.gamma - mask)
        masked_features = inputs * mask
        return masked_features, new_prior


# TabNet-like model
def tabnet_like(input_dim, hidden_dim=32, n_steps=3, gamma=1.5, dropout=0.2, l2_lambda=0.001):
    inputs = Input(shape=(input_dim,))
    x = inputs

    # Feature prior initialized as ones
    prior = tf.keras.layers.Dense(input_dim, use_bias=False,
                                    kernel_initializer='ones',
                                    trainable=False)(x)

    # Shared feature transformer layers
    shared_dense1 = Dense(hidden_dim * 2, activation=None)
    shared_dense2 = Dense(hidden_dim * 2, activation=None)

    decision_outputs = []

    for step in range(n_steps):
        # Attentive Transformer
        attn_layer = AttentiveTransformer(input_dim, l2_lambda=l2_lambda, gamma=gamma)
        masked_features, prior = attn_layer(x, prior)

        # Feature Transformer
        h = shared_dense1(masked_features)
        h = glu_block(h, hidden_dim, f"shared_glu1_{step}")

        h = shared_dense2(h)
        h = glu_block(h, hidden_dim, f"shared_glu2_{step}")

        h = Dropout(dropout)(h)
        decision_outputs.append(h)

    # Aggregate decisions and output
    aggregated = Add(name="decision_aggregation")(decision_outputs)
    outputs = Dense(1, activation="sigmoid")(aggregated)

    model = Model(inputs, outputs, name="TabNet_like")
    return model


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}

# Build model
model = tabnet_like(X_train.shape[1], hidden_dim=64, n_steps=2, dropout=0.1, gamma=1.5, l2_lambda=0.001)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[AUC(name="auroc"), AUC(curve="PR", name="auprc"),
                Precision(name="precision"), Recall(name="recall")]
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, class_weight=class_weights, verbose=1)

# Predict
y_pred = model.predict(X_test).ravel()

# Evaluate
auroc = roc_auc_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auprc = auc(recall, precision)

print(f"AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")

# Plot Precision-Recall curve
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color="blue", label="PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (TabNet-style)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tabnet_pr_curve.png", dpi=300)
plt.show()
