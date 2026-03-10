import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q)

    Parameters:
    p : array-like
        First probability distribution
    q : array-like
        Second probability distribution
    eps : float
        Small value for numerical stability

    Returns:
    float
        KL divergence
    """

    # Convert inputs to numpy arrays
    p = np.array(p)
    q = np.array(q)

    # Add epsilon to avoid log(0)
    q = q + eps

    # Handle p[i] = 0 case automatically
    kl = np.sum(p * np.log(p / q))

    return kl


# Example tests
print(kl_divergence([0.4, 0.6], [0.5, 0.5]))   # 0.0201 approx
print(kl_divergence([0.3, 0.7], [0.3, 0.7]))   # 0.0
print(kl_divergence([0.9, 0.1], [0.5, 0.5]))   # 0.368 approx