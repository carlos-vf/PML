import deepforest as df
import numpy as np
from sklearn.metrics import accuracy_score


# --- Seed setting ---
seed = 20
np.random.seed(seed)


# --- Data loading ---
X = np.load('../data/bootstrap_X.npy')
dX = np.load('../data/bootstrap_dX.npy')
y = np.load('../data/bootstrap_y.npy')

X = X[:1000]
dX = dX[:1000]
y = y[:1000]
y[y > 2] = 2

#dX = np.zeros(shape=(2000,17))
print("--- Original Data ---")
print("Unique labels in y: ", set(y))

print("--- Original Data ---")
print("Unique labels in y: ", set(y))

n_objects = X.shape[0]
n_features = X.shape[1]
n_classes = len(set(y))
print(f"{n_objects} objects, {n_features} features")


# --- Data splitting ---
n_train = int(n_objects * 0.8)
n_test =  int(n_objects - n_train)
print(f'Train set size = {n_train}, Test set size = {n_test}')

shuffled_inds = np.random.permutation(n_objects)

train_inds = shuffled_inds[:n_train]
X_train = X[train_inds]
dX_train = dX[train_inds]
y_train = y[train_inds]

test_inds = shuffled_inds[n_train:(n_train + n_test)]
X_test = X[test_inds]
dX_test = dX[test_inds]
y_test = y[test_inds]

# --- Model training ---

# 1. Regular DeepForest model for comparison
print("\n--- Fitting Regular Deep Forest for Comparison ---")
model_regular_df = df.CascadeForestClassifier(
    n_bins=n_classes,
    n_estimators=2, # Forests sets (RF + ExtraRF) per layer
    n_trees=50, # Trees per forest
    max_depth=10,
    n_jobs=1,
    random_state=seed
)
model_regular_df.fit(X_train, y_train)
y_pred_regular = model_regular_df.predict(X_test)
acc_regular = accuracy_score(y_test, y_pred_regular) * 100
print("Testing Accuracy (Regular Deep Forest): {:.3f} %".format(acc_regular))
