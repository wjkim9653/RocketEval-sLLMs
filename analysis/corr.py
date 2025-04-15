from scipy.stats import spearmanr

def spearman_correlation(dict1, dict2):
    """
    Calculate Spearman rank correlation between two dictionaries with model names as keys
    and rankings as values.

    Parameters:
    - dict1, dict2: dict[str, int or float]

    Returns:
    - Spearman correlation coefficient (float)
    """
    # Find common keys
    common_keys = set(dict1) & set(dict2)

    if len(common_keys) < 2:
        raise ValueError("Need at least two common keys to compute Spearman correlation.")

    # Extract corresponding rankings
    ranks1 = [dict1[key] for key in common_keys]
    ranks2 = [dict2[key] for key in common_keys]

    # Calculate Spearman correlation
    corr, _ = spearmanr(ranks1, ranks2)
    return corr

# Example Usage
# a = {'modelA': 1, 'modelB': 2, 'modelC': 3}
# b = {'modelA': 2, 'modelB': 3, 'modelC': 1}
# print(spearman_correlation(a, b))  # Output: -0.5