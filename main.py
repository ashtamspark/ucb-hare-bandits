"""Reproducible experiment driver for the UCB-HARE paper."""
import argparse
import hashlib
from pathlib import Path
import numpy as np

from experiments.runners import (
    run_parallel_hare,
    run_parallel_hare_policy_values,
    run_parallel_welfarist,
    run_parallel_explore_ucb,
    run_single_hare_with_policy_value,
    calculate_p_mean_regret,
    estimate_mean_reward_band,
)
from utils.plotman import (
    plot_regret_comparison,
    plot_q_ablation,
    plot_mt,
    plot_conditional_policy_value,
)
from utils.cacheman import load_or_run

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
SEED = 42
T_MAX = 1_000_000
NUM_TRIALS = 50
SIGMA_SQ = 400.0
ENV_TYPE = 1
K_ARMS = 50


def make_means(k, seed):
    return np.random.default_rng(seed).uniform(10.0, 1000.0, size=k)


def cache_name(prefix, means, horizon, trials, sigma_sq, seed):
    digest = hashlib.sha256(np.asarray(means, dtype=np.float64).tobytes()).hexdigest()[:10]
    return f"{prefix}_k{len(means)}_T{horizon}_n{trials}_s{sigma_sq:g}_{digest}_seed{seed}.npy"


def load_hare(means, horizon, trials, sigma_sq, force, seed):
    name = cache_name("hare", means, horizon, trials, sigma_sq, seed)
    return load_or_run(name, lambda: run_parallel_hare(means, horizon, sigma_sq, ENV_TYPE, trials, seed=seed), force)


def load_welfarist(means, horizon, trials, sigma_sq, p, force, seed):
    name = cache_name(f"welfarist_p{abs(p):g}", means, horizon, trials, sigma_sq, seed)
    return load_or_run(name, lambda: run_parallel_welfarist(means, horizon, sigma_sq, ENV_TYPE, p, trials, seed=seed), force)


def load_explore(means, horizon, trials, sigma_sq, p, force, seed):
    name = cache_name(f"explore_p{abs(p):g}", means, horizon, trials, sigma_sq, seed)
    return load_or_run(name, lambda: run_parallel_explore_ucb(means, horizon, sigma_sq, ENV_TYPE, p, trials, seed=seed), force)


def evaluate_algos(means, horizon, trials, force_rerun=False, seed=SEED):
    print("\nExperiment A: UCB-HARE and baselines")
    ts = np.arange(1, horizon + 1)
    arms_hare = load_hare(means, horizon, trials, SIGMA_SQ, force_rerun, seed)
    for p in (-0.5, -2.0, -20.0):
        print(f"Processing p={p}...")
        arms_wel = load_welfarist(means, horizon, trials, SIGMA_SQ, p, force_rerun, seed + 10_000)
        arms_exp = load_explore(means, horizon, trials, SIGMA_SQ, p, force_rerun, seed + 20_000)
        regrets = {
            "UCB-HARE": calculate_p_mean_regret(arms_hare, means, p)[0],
            "Welfarist UCB": calculate_p_mean_regret(arms_wel, means, p)[0],
            "Explore-Then-UCB": calculate_p_mean_regret(arms_exp, means, p)[0],
        }
        filename = f"expt_A_p_{str(abs(p)).replace('.', '_')}.png"
        plot_regret_comparison(ts, regrets, p, filename)


def evaluate_q_vary(means, horizon, trials, force_rerun=False, seed=SEED):
    print("\nExperiment B: UCB-HARE across q")
    ts = np.arange(1, horizon + 1)
    arms = load_hare(means, horizon, trials, SIGMA_SQ, force_rerun, seed)
    results = {}
    for p in (-1.0, -5.0, -10.0, -20.0):
        results[str(abs(p)).rstrip('0').rstrip('.') if p % 1 else str(int(abs(p)))] = calculate_p_mean_regret(arms, means, p)[0]
    plot_q_ablation(ts, results, "expt_B_vary_q.png")


def evaluate_mt(force_rerun=False, trials=NUM_TRIALS, seed=SEED):
    print("\nExperiment C: conditional and unconditional reward")
    means = np.array([200.0, 400.0, 600.0, 800.0, 1000.0] + [10.0] * 195)
    horizon, sigma_sq, p = 5_000, 400.0, -2.0
    ts = np.arange(1, horizon + 1)
    policy_cache = cache_name("hare_policy_values", means, horizon, trials, sigma_sq, seed + 30_000)
    hare_policy_values = load_or_run(
        policy_cache,
        lambda: run_parallel_hare_policy_values(
            means, horizon, sigma_sq, ENV_TYPE, trials, seed=seed + 30_000
        ),
        force_rerun,
    )
    arms_wel = load_welfarist(means, horizon, trials, sigma_sq, p, force_rerun, seed + 40_000)
    conditional_name = cache_name("hare_conditional", means, horizon, 1, sigma_sq, seed + 50_000)
    conditional_values = load_or_run(
        conditional_name,
        lambda: run_single_hare_with_policy_value(means, horizon, sigma_sq, ENV_TYPE, seed + 50_000)[1],
        force_rerun,
    )
    plot_conditional_policy_value(ts, conditional_values, "expt_C_conditional.png")
    mc = {
        "UCB-HARE": estimate_mean_reward_band(hare_policy_values),
        "Welfarist UCB": estimate_mean_reward_band(means[np.asarray(arms_wel, dtype=int)]),
    }
    plot_mt(ts, mc, "expt_C_mc.png")


def evaluate_d(force_rerun=False, horizon=T_MAX, trials=NUM_TRIALS, seed=SEED):
    """Optional k-ablation; saves one comparison plot for each p in {-1,-2,-5}."""
    from utils.plotman import plot_k_comparison
    print("\nExperiment D: arm-count ablation")
    for p in (-1.0, -2.0, -5.0):
        curves = {}
        for k in (10, 100):
            means = np.array([1000.0] + [950.0] * max(1, k // 10) + [10.0] * (k - 1 - max(1, k // 10)))
            np.random.default_rng(seed + k).shuffle(means)
            arms_hare = load_hare(means, horizon, trials, 1.0, force_rerun, seed + k)
            arms_wel = load_welfarist(means, horizon, trials, 1.0, p, force_rerun, seed + k + 10_000)
            curves[f"UCB-HARE, k={k}"] = calculate_p_mean_regret(arms_hare, means, p)[0]
            curves[f"Welfarist UCB, k={k}"] = calculate_p_mean_regret(arms_wel, means, p)[0]
        plot_k_comparison(np.arange(1, horizon + 1), curves, p, f"q{int(abs(p))}.png")


def main():
    parser = argparse.ArgumentParser(description="Reproduce UCB-HARE experiments")
    parser.add_argument("--expt", choices=("a", "b", "c", "d", "all"), default="all")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--horizon", type=int, default=T_MAX)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    means = make_means(K_ARMS, args.seed)
    if args.expt in ("a", "all"):
        evaluate_algos(means, args.horizon, args.trials, args.force_rerun, args.seed)
    if args.expt in ("b", "all"):
        evaluate_q_vary(means, args.horizon, args.trials, args.force_rerun, args.seed)
    if args.expt in ("c", "all"):
        evaluate_mt(args.force_rerun, args.trials, args.seed)
    if args.expt in ("d", "all"):
        evaluate_d(args.force_rerun, args.horizon, args.trials, args.seed)


if __name__ == "__main__":
    main()
