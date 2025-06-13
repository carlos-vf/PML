import PRF4DF
import probabilistic_deep_forest as pdf
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    # --- Load Data ---
    X = np.load('data/bootstrap_X.npy')
    dX_full = np.load('data/bootstrap_dX.npy') # Use a different name to avoid confusion
    y = np.load('data/bootstrap_y.npy')
    y[y > 2] = 2

    print("X: ", X)
    print("Xd: ", dX_full)
    print("y: ", y)
    print("labels: ", set([int(i) for i in y]))

    n_objects = X.shape[0]
    n_features = X.shape[1]

    # --- Data Splitting ---
    shuffled_inds = np.random.permutation(n_objects)
    
    n_train = 500
    n_test = 50
    print('Train set size = {}, Test set size = {}'.format(n_train, n_test))

    train_inds = shuffled_inds[:n_train]
    X_train = X[train_inds]
    
    # CRITICAL BUG FIX: Slice the dX array, not the X array, for uncertainty.
    dX_train = dX_full[train_inds] 
    
    y_train = y[train_inds]

    test_inds = shuffled_inds[n_train:(n_train + n_test)]
    X_test = X[test_inds]
    y_test = y[test_inds]

    # For the test set, we can pass dX=None if it's considered "clean"
    # or slice it as well if it has known uncertainty.
    dX_test = dX_full[test_inds]

    # --- Model Training ---
    n_cascade_estimators = 4
    n_trees_pdf = 20
    max_depth_pdf = None
    accuracies_pdf = []
    
    # We will use one seed for this test for simplicity
    seed = 42
    
    model = pdf.CascadeForestClassifier(random_state=seed)
    
    prf_estimators = []
    for i in range(n_cascade_estimators):
        estimator = PRF4DF.SklearnCompatiblePRF(
            n_classes_=3,
            n_features_=n_features,
            n_estimators=n_trees_pdf,
            max_depth=max_depth_pdf,
            n_jobs=-1
        )
        prf_estimators.append(estimator)
        
    model.set_estimator(prf_estimators)
    
    print("Fitting model...")
    # Pass the correctly sliced dX_train
    model.fit(X=X_train, y=y_train, dX=dX_train)
    print("Model fitting complete.")
    
    print("Predicting on test set...")
    # Pass the test uncertainty array to predict
    y_pred = model.predict(X_test, dX=dX_test)
    acc = accuracy_score(y_pred, y_test)
    accuracies_pdf.append(acc)
    
    print("Mean Accuracy: ", np.mean(accuracies_pdf))


if __name__ == "__main__":
    main()