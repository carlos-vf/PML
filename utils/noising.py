import numpy as np

def add_noise(X, noise_type='gaussian',
              noise_scale=0.1,
              beta_alpha=2.0,
              beta_beta=5.0,
              beta_scale=None,
              apply_to_test=False,
              X_test=None,
              random_seed=27):
    """
    Add noise to dataset X (features). Optionally also add noise to X_test.

    Parameters:
    - X: np.array shape (n_samples, n_features)
    - noise_type: str, one of ['gaussian', 'beta', 'gaussian_heteroschedastic', 'uniform']
    - noise_scale: float, scale for Gaussian noise mean and std dev (for 'gaussian'and 'uniform' only)
    - beta_alpha, beta_beta: float, Beta distribution parameters (for 'beta' only)
    - beta_scale: float or None, scaling for Beta noise (default 0.2 if None)
    - apply_to_test: bool, whether to apply noise to X_test as well
    - X_test: np.array, test set to apply noise if apply_to_test=True
    - random_seed: int, random seed for reproducibility

    Returns:
    - X_noisy: np.array, noisy version of X
    - X_test_noisy: np.array or None, noisy version of X_test if apply_to_test else None
    - dX: np.array, noise added to X
    - dX_test: np.array or None, noise added to X_test if apply_to_test else None
    """
    np.random.seed(random_seed)

    n, n_features = X.shape
    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)

    def generate_noise(n_samples, X_sample):
        if noise_type == 'gaussian':
            # 1. Sample per-feature Beta coefficients and normalize to mean 1
            beta_per_feature = np.random.beta(1.1, 1, size=n_features)
            beta_per_feature /= beta_per_feature.mean()  # Normalize to mean=1, so it's coherent with the noise_scale parameter

            # 2. Expand to shape (n_samples, n_features)
            coeffs = np.tile(beta_per_feature, (n_samples, 1))  # Shape: (n_samples, n_features)

            # 3. Compute per-element noise variance
            noise_vars = noise_scale * (feature_stds ** 2)[np.newaxis, :] * coeffs

            # 4. Sample noise with zero mean and computed std
            noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_vars))

            return noise


        elif noise_type == 'beta':
            _beta_scale = beta_scale if beta_scale is not None else 0.2
            beta_noise = np.random.beta(beta_alpha, beta_beta, size=(n_samples, n_features))
            beta_mean = beta_alpha / (beta_alpha + beta_beta)
            centered = beta_noise - beta_mean
            return centered * _beta_scale * feature_stds

        elif noise_type == 'gaussian_heteroschedastic':
            epsilon = 1e-6
            local_std = noise_scale * (np.abs(X_sample) + epsilon)
            return np.random.normal(loc=0, scale=local_std)

        elif noise_type == 'uniform':
            noise_low = -noise_scale * feature_stds
            noise_high = noise_scale * feature_stds
            noise = np.random.uniform(low=noise_low, high=noise_high, size=(n_samples, n_features))
            return noise

        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

    dX = generate_noise(n, X)
    X_train_noisy = X + dX

    dX_test = None
    X_test_noisy = None
    if apply_to_test:
        if X_test is None:
            raise ValueError("X_test must be provided if apply_to_test=True")
        n_test = X_test.shape[0]
        dX_test = generate_noise(n_test, X_test)
        X_test_noisy = X_test + dX_test

    return X_train_noisy, X_test_noisy, dX, dX_test


import numpy as np

def add_label_noise(
    y,
    apply_to_test=False,
    y_test=None,
    noise_level=0.1,
    mode="proportion",  # "proportion" or "random_prob"
    prob_range=(0.0, 0.5),
    random_seed=27
):
    """
    Add noise to labels y, either by flipping a fixed proportion (mode='proportion')
    or by assigning each sample a noise probability from a uniform distribution (mode='random_prob').

    Returns:
    - y_noisy: np.array shape (n_samples,), noisy labels
    - y_test_noisy: np.array or None, noisy test labels if apply_to_test else None
    - probs_y: np.array shape (n_samples, n_classes), label probability distribution per sample
    - probs_y_test: np.array or None, same for test samples if apply_to_test else None
    """

    if not (0 <= noise_level <= 1):
        raise ValueError("noise_level must be between 0 and 1.")

    if mode == "random_prob":
        if not (0 <= prob_range[0] <= 1) or not (0 <= prob_range[1] <= 1):
            raise ValueError("prob_range values must be between 0 and 1.")
        if prob_range[0] > prob_range[1]:
            raise ValueError("prob_range[0] must be <= prob_range[1].")

    np.random.seed(random_seed)
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Helper: build probability distribution for a single sample
    def build_prob_dist(true_label, flip_prob):
        # Handle the case where n_classes is 1 (no other classes to distribute flip_prob to)
        if n_classes - 1 == 0:
            prob_dist = np.full(n_classes, 0.0) # No other classes, so probability is 0 for non-existent ones
        else:
            prob_dist = np.full(n_classes, flip_prob / (n_classes - 1))
        true_idx = label_to_idx[true_label]
        prob_dist[true_idx] = 1 - flip_prob
        return prob_dist

    y_noisy = y.copy()
    probs_y = np.zeros((len(y), n_classes))

    if mode == "proportion":
        # Calculate the number of samples to actually flip in y_noisy
        n_flip = int(noise_level * len(y))
        # Randomly select the indices of samples whose labels will be flipped in y_noisy
        indices_to_flip_in_y_noisy = np.random.choice(len(y), size=n_flip, replace=False)

        for idx in range(len(y)):
            
            true_label = y[idx]

            # FOR probs_y: Every sample has the *same* intrinsic 'noise_level' chance of switching,
            # regardless of whether it's actually flipped in y_noisy.
            # This is the probability distribution reflecting the "game rules" for all labels.
            probs_y[idx] = build_prob_dist(true_label, noise_level) # <-- ALWAYS use noise_level here for probs_y

            # FOR y_noisy: Only flip the label if this sample's index was chosen
            if idx in indices_to_flip_in_y_noisy:
                current_label = y_noisy[idx]
                # Select a new label that is not the current one
                # Handle cases where current_label is the only unique_label (shouldn't happen with n_classes > 1 but for robustness)
                possible_new_labels = [l for l in unique_labels if l != current_label]
                if possible_new_labels: # Ensure there's a label to switch to
                    new_label = np.random.choice(possible_new_labels)
                    y_noisy[idx] = new_label
                # else: if there's only one class, no flipping is possible.

    elif mode == "random_prob":
        min_p, max_p = prob_range
        flip_probs = np.random.uniform(min_p, max_p, size=len(y))
        for idx, flip_prob in enumerate(flip_probs):
            true_label = y[idx]
            probs_y[idx] = build_prob_dist(true_label, flip_prob)
            if np.random.rand() < flip_prob:
                current_label = y_noisy[idx]
                possible_new_labels = [l for l in unique_labels if l != current_label]
                if possible_new_labels:
                    new_label = np.random.choice(possible_new_labels)
                    y_noisy[idx] = new_label
    else:
        raise ValueError("mode must be either 'proportion' or 'random_prob'")

    y_test_noisy = None
    probs_y_test = None
    if apply_to_test:
        if y_test is None:
            raise ValueError("y_test must be provided if apply_to_test=True")
        y_test_noisy = y_test.copy()
        probs_y_test = np.zeros((len(y_test), n_classes))

        # Test set logic mirroring the training set
        if mode == "proportion":
            n_flip_test = int(noise_level * len(y_test))
            indices_to_flip_in_y_test_noisy = np.random.choice(len(y_test), size=n_flip_test, replace=False)
            for idx in range(len(y_test)):
                true_label = y_test[idx]
                # FOR probs_y_test: Every sample has the *same* intrinsic 'noise_level' chance of switching
                probs_y_test[idx] = build_prob_dist(true_label, noise_level) # <-- ALWAYS use noise_level here for probs_y_test

                # FOR y_test_noisy: Only flip the label if this sample's index was chosen
                if idx in indices_to_flip_in_y_test_noisy:
                    current_label = y_test_noisy[idx]
                    possible_new_labels = [l for l in unique_labels if l != current_label]
                    if possible_new_labels:
                        new_label = np.random.choice(possible_new_labels)
                        y_test_noisy[idx] = new_label

        elif mode == "random_prob":
            min_p, max_p = prob_range
            flip_probs = np.random.uniform(min_p, max_p, size=len(y))
            for idx, flip_prob in enumerate(flip_probs):
                true_label = y[idx]
                probs_y[idx] = build_prob_dist(true_label, flip_prob)
                # Remove the explicit flip - it's already handled in the probability distribution
                # The noisy label will be sampled from this distribution later if needed
                y_noisy[idx] = true_label  # Keep original label by default
    
    return y_noisy, y_test_noisy, probs_y, probs_y_test