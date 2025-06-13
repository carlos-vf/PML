import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters
a, b = 1.1,1.0

# Generate samples
np.random.seed(42)
samples = np.random.beta(a, b, 10)
your_examples = [0.7, 0.6, 0.98, 0.4, 0.8, 0.2, 0.1, 0.7, 0.89]

# Plot
plt.figure(figsize=(10, 5))

# Plot density curve
x = np.linspace(0, 1, 1000)
plt.plot(x, beta.pdf(x, a, b), 'k-', lw=2, label=f'Beta({a},{b}) Density')

# Plot generated samples (jittered for visibility)
plt.scatter(samples, -0.01 * np.ones_like(samples), 
            c='blue', alpha=0.5, s=50, label='Generated Samples (n=100)')

plt.title(f'Beta({a},{b}) Distribution vs. Samples')
plt.xlabel('Value')
plt.yticks([], [])
plt.ylim(-0.05, 3)
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()