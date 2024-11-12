import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.metrics.pairwise import rbf_kernel


def eval_metrics(trajectory, X, y, t, guidance, train, kl_div_skip=False):
    """Prepare tensors for evaluation and compute metrics.

    Args:
        trajectory (np.ndarray): Trajectory of the samples. Shape (n_samples, n_timepoints, n_features).
        X (np.ndarray): Ground truth/reference data
        y (np.ndarray): Ground truth/reference labels.
        t (np.ndarray): Time points.
        guidance (float): Guidance strength used for sampling the trajectory.
        train (bool): Whether the data is training or testing.
        kl_div_skip (bool): Whether to skip computing the KL divergence.
    """
    # Permute first two dimensions to match shape of trajectory to X_{train, valid}
    # This allows easier alignment of the tensors below
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory[np.arange(trajectory.shape[0])[:, None], (t * (trajectory.shape[1] - 1)).astype(int)]

    # We loop over all marginals (timepoints) and conditions to compute the metrics
    results = []
    first_unique_c = np.unique(y[:, 0])
    for marginal in range(trajectory.shape[1]):
        marginal_unique_c = sorted({x.item() for x in np.unique(y[:, marginal]) if np.isfinite(x)})
        for c in marginal_unique_c:
            if c in first_unique_c:
                target = X[:, marginal][y[:, marginal] == c]
                transport = trajectory[:, marginal][y[:, 0] == c]
                mmd, mmd_median, wasserstein, mean_diff_l1, mean_diff_l2, kl_div = compute_metric_set(
                    target, transport, kl_div_skip=kl_div_skip
                )
            else:
                mmd = np.nan
                mmd_median = np.nan
                wasserstein = np.nan
                mean_diff_l1 = np.nan
                mean_diff_l2 = np.nan
                kl_div = np.nan

            results.append(
                {
                    "marginal": marginal,
                    "c": c,
                    "mmd": mmd,
                    "mmd_median": mmd_median,
                    "wasserstein": wasserstein,
                    "guidance": guidance,
                    "train": train,
                    "time": t[0, marginal],
                    "mean_diff_l1": mean_diff_l1,
                    "mean_diff_l2": mean_diff_l2,
                    "kl_div": kl_div,
                }
            )
    return pd.DataFrame(results)


def compute_metric_set(target, transport, kl_div_skip=False):
    """Compute a set of metrics between target and transport distributions.

    Args:
        target (np.ndarray): Target distribution(s).
        transport (np.ndarray): Transport distribution(s).
        kl_div_skip (bool): Whether to skip computing the KL divergence.
    """
    mmd = compute_scalar_mmd(target, transport)
    mmd_median = compute_scalar_mmd(target, transport, use_median_heuristic=True)
    wasserstein = wasserstein_loss(target, transport).item()
    # Compute difference in the means of the target and transport distribution
    mean_diff_l1 = np.linalg.norm(np.mean(target, axis=0) - np.mean(transport, axis=0), ord=1)
    mean_diff_l2 = np.linalg.norm(np.mean(target, axis=0) - np.mean(transport, axis=0), ord=2)
    if not kl_div_skip:
        kl_div = kl_divergence_under_gaussian(target, transport)
    else:
        kl_div = np.nan

    return mmd, mmd_median, wasserstein, mean_diff_l1, mean_diff_l2, kl_div


def kl_divergence_under_gaussian(a, b):
    """Compute the KL divergence between two Gaussian distributions."""
    mean_a = np.mean(a, axis=0)
    covariance_a = np.cov(a, rowvar=False)

    mean_b = np.mean(b, axis=0)
    covariance_b = np.cov(b, rowvar=False)

    def safe_kld(mean_a, covariance_a, mean_b, covariance_b):
        try:
            kld = 0.5 * (
                np.log(np.linalg.det(covariance_b) / np.linalg.det(covariance_a))
                + np.trace(
                    np.linalg.inv(covariance_b) @ covariance_a
                    + (mean_b - mean_a).T @ np.linalg.inv(covariance_b) @ (mean_b - mean_a)
                )
                - a.shape[1]
            )
        except ValueError:
            kld = np.nan
        return kld

    return safe_kld(mean_a, covariance_a, mean_b, covariance_b)


def compute_scalar_mmd(target, transport, gammas=None, use_median_heuristic=False):
    """Compute the MMD distance between two sets of samples.

    Adapted from: https://github.com/bunnech/condot/tree/main/condot/losses

    Args:
        target (np.ndarray): Target distribution(s).
        transport (np.ndarray): Transport distribution(s).
        gammas (list): List of gamma values to use for the MMD computation.
        use_median_heuristic (bool): Whether to use the median heuristic for gamma.
    """
    if gammas is not None and use_median_heuristic:
        raise ValueError("Cannot use both gammas and median heuristic.")
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    if not isinstance(gammas, list):
        gammas = [gammas]
    if use_median_heuristic:
        if target.shape[0] != transport.shape[0]:
            median = np.median(np.linalg.norm(np.unique(target, axis=0) - transport, axis=1) ** 2)
        else:
            median = np.median(np.linalg.norm(target - transport, axis=1) ** 2)
        gammas = [median]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def mmd_distance(x, y, gamma):
    """Compute the MMD distance between two sets of samples."""
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def wasserstein_loss(x, y, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm.

    Adapted from: https://github.com/bunnech/condot/tree/main/condot/losses
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon)

    try:
        return loss(x, y)
    except ValueError:
        return torch.tensor(np.nan)
