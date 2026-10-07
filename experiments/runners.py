"""Simulation runners and welfare estimators for the paper experiments."""
import numpy as np
from statistics import NormalDist
from joblib import Parallel, delayed

from algorithms.hare import simulate_hare
from algorithms.welfarist import simulate_welfarist
from algorithms.explore_ucb import simulate_explore_ucb


def run_parallel_hare(means, T, sigma_sq, env_type, num_trials, n_jobs=-1, seed=42):
    """Return chosen-arm histories for each independent run."""
    print(f"Spawning {num_trials} parallel HARE instances...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_hare)(means, T, sigma_sq, env_type, seed + i)
        for i in range(num_trials)
    )
    return np.vstack([result[0] for result in results])


def run_parallel_welfarist(means, T, sigma_sq, env_type, p, num_trials, n_jobs=-1, seed=42):
    print(f"Spawning {num_trials} parallel Welfarist instances...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_welfarist)(means, T, sigma_sq, env_type, p, seed + i)
        for i in range(num_trials)
    )
    return np.vstack(results)


def run_parallel_explore_ucb(means, T, sigma_sq, env_type, p, num_trials, n_jobs=-1, seed=42):
    print(f"Spawning {num_trials} parallel Explore-Then-UCB instances...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_explore_ucb)(means, T, sigma_sq, env_type, p, seed + i)
        for i in range(num_trials)
    )
    return np.vstack(results)


def _cumulative_p_mean(values, p):
    """Cumulative p-mean over time for one or more reward sequences."""
    values = np.asarray(values, dtype=float)
    T = values.shape[-1]
    counts = np.arange(1, T + 1, dtype=float)
    if p == 0:
        return np.exp(np.cumsum(np.log(np.maximum(values, 1e-300)), axis=-1) / counts)
    if p == 1:
        return np.cumsum(values, axis=-1) / counts
    if p < 0:
        positive = values > 0
        log_values = p * np.log(np.where(positive, values, 1.0))
        log_values = np.where(positive, log_values, np.inf)
        log_sums = np.logaddexp.accumulate(log_values, axis=-1)
        return np.exp((log_sums - np.log(counts)) / p)
    return np.power(np.cumsum(np.power(values, p), axis=-1) / counts, 1.0 / p)


def calculate_p_mean_regret(arms_matrix, means, p):
    """Estimate ex-ante m_t across runs, then compute cumulative p-mean regret."""
    arms_matrix = np.asarray(arms_matrix, dtype=int)
    trial_rewards = np.asarray(means, dtype=float)[arms_matrix]
    m_t = trial_rewards.mean(axis=0)
    welfare = _cumulative_p_mean(m_t, p)
    return float(np.max(means)) - welfare, m_t


def estimate_mean_reward_band(run_traces, confidence=0.95):
    """Estimate m_t and pointwise normal intervals from independent run-level m_t traces."""
    run_traces = np.asarray(run_traces, dtype=float)
    center = run_traces.mean(axis=0)
    if run_traces.shape[0] < 2:
        return center, center.copy(), center.copy()
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    half_width = z * run_traces.std(axis=0, ddof=1) / np.sqrt(run_traces.shape[0])
    return center, np.maximum(0.0, center - half_width), center + half_width



