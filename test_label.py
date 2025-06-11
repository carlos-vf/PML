# Test case
import numpy as np
from utils.noising import add_label_noise
y = np.array([0, 0, 0, 1, 1,2,4,4,4,3,3,1,3, 1])
noise_level = 0.3
prob_range = (0.2, 0.4)  # 20-40% flip probability

# Run original and corrected versions
y_noisy_orig, _, probs_y_orig, _ = add_label_noise(y, noise_level=noise_level, mode="random_prob", prob_range=prob_range, random_seed=1)

# Compare flip rates
orig_flip_rate = np.mean(y_noisy_orig != y)
print(f"Original flip rate: {orig_flip_rate:.2f} (higher than expected)")

# head of y
print("Original labels:", y[:10])

# Corrected version with adjusted noise level
print("Original noisy labels:", y_noisy_orig[:10])

# probs_y_orig printing
print("Original label probabilities:", probs_y_orig[:10])
