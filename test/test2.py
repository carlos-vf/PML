from deepforest import CascadeForestClassifier
import PRF
import numpy as np


X = np.load('../data/bootstrap_X.npy')
dX = np.load('../data/bootstrap_dX.npy')
y = np.load('../data/bootstrap_y.npy')


print("Original X sample: ", X[:2])
print("Original dX sample: ", dX[:2])
print("Original y sample: ", y[:5])
print("Unique labels in y: ", set([int(i) for i in y]))

n_objects = X.shape[0]
n_features = X.shape[1]
n_classes = len(set(y))
print(f"{n_objects} objects, {n_features} features")

n_train = min(5000, int(n_objects * 0.8))
n_test = min(500, n_objects - n_train)
print(f'Train set size = {n_train}, Test set size = {n_test}')

# Seed for reproducibility of shuffle
np.random.seed(42)
shuffled_inds = np.random.permutation(n_objects) # Use permutation for no replacement by default

train_inds = shuffled_inds[:n_train]
X_train = X[train_inds][:, :n_features]
dX_train = dX[train_inds][:, :n_features] 
y_train = y[train_inds]

test_inds = shuffled_inds[n_train:(n_train + n_test)]
X_test = X[test_inds][:, :n_features]
dX_test = dX[test_inds][:, :n_features]
y_test = y[test_inds]

# Concatenate X and dX for training
X_train_combined = np.hstack((X_train, dX_train))
X_test_combined = np.hstack((X_test, dX_test))
print(X_train_combined)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# --- Model training ---
# PRFs
n_cascade_estimators = 4
prf_estimators = []
for i in range(n_cascade_estimators):
    single_prf_estimator = PRF.SklearnCompatiblePRF(
        n_classes_= n_classes,
        n_features_= n_features,
        n_estimators=20,
        max_depth=10,
        random_state=i,
        n_jobs=1
    )
    prf_estimators.append(single_prf_estimator)

# DeepForest
model = CascadeForestClassifier(
    n_bins=n_classes,
    random_state=42,
    n_estimators=4,
    n_trees=20,
)
model.set_estimator(prf_estimators)

print("Starting model fitting...")
model.fit(X=X_train_combined, y=y_train)
#model.fit(X=X_train, y=y_train)
print("Model fitting finished.")

# --- Prediction and Evaluation ---
# y_pred_proba = model.predict_proba(X_test) # Not used directly for score
# y_pred = model.predict(X_test) # Not used directly for score

accuracy = model.score(X_test_combined, y_test) * 100
print(f"\nTesting Accuracy: {accuracy:.3f} %")