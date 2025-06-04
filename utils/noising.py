import numpy as np

def add_noise(X, noise_type='gaussian',
              gaussian_scale=0.1,
              beta_alpha=2.0,
              beta_beta=5.0,
              beta_scale=None,
              uniform_scale=0.1,
              apply_to_test=False,
              X_test=None,
              random_seed=27):
    """
    Add noise to dataset X (features). Optionally also add noise to X_test.

    Parameters:
    - X: np.array shape (n_samples, n_features)
    - noise_type: str, one of ['gaussian', 'beta', 'gaussian_heteroschedastic', 'uniform']
    - gaussian_scale: float, scale for Gaussian noise mean and std dev (for 'gaussian' only)
    - beta_alpha, beta_beta: float, Beta distribution parameters (for 'beta' only)
    - beta_scale: float or None, scaling for Beta noise (default 0.2 if None)
    - uniform_scale: float, scaling for uniform noise range (for 'uniform' only)
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
            noise_means = np.random.uniform(-gaussian_scale * feature_means,
                                            gaussian_scale * feature_means,
                                            size=(n_samples, n_features))
            noise_vars = (gaussian_scale * feature_stds) ** 2
            noise_vars = np.tile(noise_vars, (n_samples, 1))
            return np.random.normal(loc=noise_means, scale=np.sqrt(noise_vars))

        elif noise_type == 'beta':
            _beta_scale = beta_scale if beta_scale is not None else 0.2
            beta_noise = np.random.beta(beta_alpha, beta_beta, size=(n_samples, n_features))
            beta_mean = beta_alpha / (beta_alpha + beta_beta)
            centered = beta_noise - beta_mean
            return centered * _beta_scale * feature_stds

        elif noise_type == 'gaussian_heteroschedastic':
            epsilon = 1e-6
            local_std = gaussian_scale * (np.abs(X_sample) + epsilon)
            return np.random.normal(loc=0, scale=local_std)

        elif noise_type == 'uniform':
            noise_low = -uniform_scale * feature_stds
            noise_high = uniform_scale * feature_stds
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
