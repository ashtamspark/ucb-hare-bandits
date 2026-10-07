# UCB-HARE experiments

This directory contains the algorithms and experiment runner used by the accompanying AISTATS paper. It uses NumPy, Numba, Joblib, and Matplotlib. Install with `python -m pip install -r requirements.txt`.

Run all experiments from this directory with A, B, and D at one million rounds and 50 runs, and C at 5,000 rounds and 1,000 runs:

```powershell
python main.py --expt all
```

The same command can be launched from the parent directory as `python ucb-hare-bandits/main.py --expt all`; caches and plots are saved beside the code in either case. To redraw only the revised C and D figures from available caches, run `python main.py --expt c` and `python main.py --expt d`. Experiment C saves separate `expt_C_mc.png` and `expt_C_smoothed.png` plots, and copies both into the paper's `figures/` directory when run inside the AISTATS package. The second plot averages each run over the current and preceding nine rounds before computing the across-run mean and confidence intervals. To update the D paper assets, copy `results/q1.png`, `q2.png`, and `q5.png` into `../UCB-HARE-AISTATS/figures/` before compiling the paper.

Run Experiment C with `python main.py --expt c` (1,000 independent runs by default). Use `--c-trials R` to choose a different number of runs. The CLI accepts `a`, `b`, `c`, `d`, or `all`; `--trials` controls A, B, and D, `--force-rerun` ignores saved simulation matrices, and `--seed` sets the deterministic seed sequence. Results are written to `results/`, and simulation matrices are cached under `cached_data/`. Cache keys include the horizon, trial count, reward variance, mean vector, and seed.

Experiments A and B use 50 Gaussian arms with means drawn uniformly from `[10, 1000]`, variance 400, and horizon 1,000,000. UCB-HARE is `p`-agnostic: Experiment B reuses the same HARE action traces for all plotted values of `p`. Experiment C uses 200 arms, five elevated means, variance 400, and horizon 5,000. On independent run `r`, let `m_tilde_t^(r)` be the run-level expected arm mean: UCB-HARE averages over the hidden within-block ordering choice given that run's ordering and observed history; Welfarist-UCB uses the true mean of the selected arm. The target is `m_t = E[m_tilde_t]`, with expectation over a fresh run. Experiment C uses `R=1,000` independent runs by default (set with `--c-trials`) and saves two separate plots: the unsmoothed estimate `m_hat_t = (1/R) sum_r m_tilde_t^(r)` with pointwise 95% normal confidence intervals, and its 10-round trailing average with intervals computed across the run-level 10-round averages. Experiment D varies the arm count and uses variance 1.

The A, B, and D plots show point estimates of p-mean regret from the estimated unconditional `m_t` sequence and omit ribbons. In D, dashed lines denote `k=10` and solid lines denote `k=100`. Experiment C uses a thin line for the across-run estimate and a translucent pointwise 95% normal confidence interval, computed as the sample mean plus or minus 1.96 standard errors and clipped below at zero. The intervals describe uncertainty at each round separately, not simultaneous coverage over the full time series. The HARE simulator uses the paper's `delta = 1/T` setting and checks the stopping condition only at block boundaries.
