# UCB-HARE experiments

This directory contains the algorithms and experiment runner used by the accompanying AISTATS paper. It uses NumPy, Numba, Joblib, and Matplotlib. Install with `python -m pip install -r requirements.txt`.

Run all experiments at the paper settings (one million rounds, 50 runs) from this directory with:

```powershell
python main.py --expt all
```

The same command can be launched from the parent directory as `python ucb-hare-bandits/main.py --expt all`; caches and plots are saved beside the code in either case.

Run a smoke-sized experiment with `python main.py --expt c --horizon 5000 --trials 10`. The CLI accepts `a`, `b`, `c`, `d`, or `all`; `--force-rerun` ignores saved simulation matrices, and `--seed` sets the deterministic seed sequence. Results are written to `results/`, and simulation matrices are cached under `cached_data/`. Cache keys include the horizon, trial count, reward variance, mean vector, and seed.

Experiments A and B use 50 Gaussian arms with means drawn uniformly from `[10, 1000]`, variance 400, and horizon 1,000,000. UCB-HARE is `p`-agnostic: Experiment B reuses the same HARE action traces for all plotted values of `p`. Experiment C uses 200 arms, five elevated means, variance 400, and horizon 5,000. It plots one run's unsmoothed conditional policy values `tilde_m_t = E[mu_{I_t} | pi, H_{t-1}]` and estimates `m_t = E[mu_{I_t}]` by averaging those conditional values across independent HARE runs. For Welfarist-UCB, `m_t` is estimated by averaging selected arm means across runs. The right-panel intervals are pointwise 95% normal Monte Carlo intervals across runs. Experiment D varies the arm count and uses variance 1.

The A, B, and D plots show point estimates of p-mean regret from the estimated unconditional `m_t` sequence and omit ribbons. Experiment C's pointwise intervals describe uncertainty at each round separately; they are not simultaneous confidence bands over the full time series. The HARE simulator uses the paper's `delta = 1/T` setting and checks the stopping condition only at block boundaries.
