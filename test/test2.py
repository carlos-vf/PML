import deepforest as df
import os
import PRF4DF
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import PRF

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

# DeepForest with Custom PRF Estimators
model_custom_prf = df.CascadeForestClassifier(
    n_bins=n_classes,
    random_state=seed,
)

# PRF (estimators)
prf_estimators = []
for i in range(4):
    single_prf_estimator = PRF4DF.SklearnCompatiblePRF(
        n_classes_=n_classes,
        n_features_=n_features,
        n_estimators=100, # Trees per forest
        max_depth=10,
        n_jobs=1
    )
    prf_estimators.append(single_prf_estimator)

# Set the PRF estimators to the DF model
model_custom_prf.set_estimator(prf_estimators)

# --- Model fitting ---
print("Starting model fitting with Custom PRF...")
model_custom_prf.fit(X=X_train, dX=dX_train, y=y_train)

# --- Model evaluation ---
y_pred_custom = model_custom_prf.predict(X=X_test, dX=dX_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom) * 100
print(f"Testing Accuracy (Deep Forest with Custom PRF): {accuracy_custom:.3f} %")



if model_custom_prf.n_layers_ > 1:
    print("\n--- [VERIFICATION] Analyzing Feature Importances of Layer 1 ---")
    
    # Get the second layer (index 1)
    layer_1 = model_custom_prf.get_layer(1)
    
    # Get the first estimator from that layer
    # The key format is "{layer_idx}-{estimator_idx}-custom"
    first_estimator_in_layer_1 = layer_1.estimators_['1-0-custom']
    
    # This estimator is a KFoldWrapper. We need to average its importances.
    importances = np.mean([
        est.feature_importances_ for est in first_estimator_in_layer_1.estimators_
    ], axis=0)

    n_original_features = X_train.shape[1]
    original_feat_importance = np.sum(importances[:n_original_features])
    augmented_feat_importance = np.sum(importances[n_original_features:])
    
    print(f"Total importance of ORIGINAL features: {original_feat_importance:.4f}")
    print(f"Total importance of AUGMENTED features: {augmented_feat_importance:.4f}")

    if augmented_feat_importance < 0.01:
        print("\nWARNING: The augmented features have very low importance. The model is likely ignoring them.")

# --- Comparison Models ---

""" # 2. Regular PRF model for comparison
print("\n--- Fitting Standalone PRF for Comparison ---")
model_standalone_prf = PRF4DF.prf(
    n_classes_=n_classes,
    n_features_=n_features,
    n_estimators=50,
    max_depth=10,
    n_jobs=1,
)
model_standalone_prf.fit(X=X_train, y=y_train, dX=dX_train)
y_pred_standalone = model_standalone_prf.predict(X=X_test, dX=dX_test)
acc_standalone = accuracy_score(y_test, y_pred_standalone) * 100
print(f"Testing Accuracy (Standalone PRF): {acc_standalone:.3f} %")


print("\n--- Fitting Standalone PRF for Comparison ---")
model_standalone_prf = PRF.prf(
    n_estimators=50,
    max_depth=10,
    n_jobs=1,
)
model_standalone_prf.fit(X=X_train, y=y_train, dX=dX_train)
y_pred_standalone = model_standalone_prf.predict(X=X_test, dX=dX_test)
acc_standalone = accuracy_score(y_test, y_pred_standalone) * 100
print(f"Testing Accuracy (Standalone PRF): {acc_standalone:.3f} %")
 """
