"""Publication plots in the paper's existing blue/orange/green palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

COLORS = {"UCB-HARE": "blue", "Welfarist UCB": "orange", "Explore-Then-UCB": "green"}
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _downsample(ts, arrays, num_points=10000):
    """Downsample aligned curves, preserving arrays and confidence-band tuples."""
    ts = np.asarray(ts)
    if ts.ndim != 1:
        raise ValueError(f"ts must be one-dimensional; received shape {ts.shape}")

    normalized = {}
    for name, value in arrays.items():
        is_tuple = isinstance(value, tuple)
        parts = value if is_tuple else (value,)
        converted = tuple(np.asarray(part) for part in parts)
        for part in converted:
            if part.ndim != 1 or part.shape[0] != ts.shape[0]:
                raise ValueError(
                    f"curve {name!r} must be one-dimensional and match ts "
                    f"(length {ts.shape[0]}); received shape {part.shape}"
                )
        normalized[name] = converted if is_tuple else converted[0]

    if ts.size <= num_points:
        return ts, normalized

    idx = np.linspace(0, ts.size - 1, num_points, dtype=int)
    sampled = {
        name: tuple(part[idx] for part in value) if isinstance(value, tuple) else value[idx]
        for name, value in normalized.items()
    }
    return ts[idx], sampled


def _save(fig, filename):
    fig.savefig(RESULTS_DIR / filename, dpi=300)


def _band(value):
    if isinstance(value, tuple) and len(value) == 3:
        return value
    y = np.asarray(value)
    return y, y, y


def _draw_bands(ax, ts, data, labels=None):
    series = []
    for name, value in data.items():
        y, lower, upper = _band(value)
        color = COLORS.get(name, None)
        label = labels.get(name, name) if labels else name
        if np.any(np.asarray(upper) > np.asarray(lower)):
            ax.fill_between(
                ts, lower, upper, color=color, alpha=0.14, linewidth=0,
                label="_nolegend_", zorder=1,
            )
        series.append((name, y, color, label))
    for name, y, color, label in series:
        ax.plot(
            ts, y, label=label, linewidth=1.35 if name == "UCB-HARE" else 1.2,
            color=color, zorder=3 if name == "UCB-HARE" else 2,
        )


def plot_regret_comparison(ts, regrets_dict, p, filename):
    ts, data = _downsample(ts, regrets_dict)
    mask = ts >= 1e2
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for name, value in data.items():
        color = COLORS.get(name, None)
        ax.plot(ts[mask], np.asarray(value)[mask], label=name,
                linewidth=2.2 if name == "UCB-HARE" else 1.9,
                color=color, zorder=3 if name == "UCB-HARE" else 2)
    ax.set(xlabel='Round $t$ (log scale)', ylabel=r"$p$-mean regret", title=rf"$p = {p}$")
    ax.set_xscale('log')
    ax.legend(loc='best', frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    plt.close(fig)


def plot_q_ablation(ts, regrets_dict, filename):
    ts, data = _downsample(ts, regrets_dict)
    mask = ts >= 1e2
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = ["blue", "orange", "green", "red", "purple"]
    for i, (p_val, value) in enumerate(data.items()):
        y = np.asarray(value)
        color = colors[i % len(colors)]
        ax.plot(ts[mask], y[mask], label=rf"UCB-HARE ($p=-{p_val}$)", linewidth=2, color=color)
    ax.set(xlabel='Round $t$ (log scale)', ylabel=r"$p$-mean regret", title=r"UCB-HARE ($p$-agnostic) across fairness levels")
    ax.set_xscale('log')
    ax.legend(loc='best', frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    plt.close(fig)


def plot_mt(ts, mt_dict, filename="expt_C_mc.png"):
    """Plot the across-run estimate of m_t with pointwise Monte Carlo intervals."""
    ts, data = _downsample(ts, mt_dict)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _draw_bands(ax, ts, data)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Estimate $\widehat{m}_t$")
    ax.set_title(r"Monte Carlo estimate of $m_t$")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    plt.close(fig)

def plot_regret_vs_time_multi_k(results_dict, p, filename="regret_vs_time.png"):
    """Plot regret by algorithm and arm count using color and line style."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    base_colors = {"UCB-HARE": "#1f77b4", "Welfarist UCB": "#ff7f0e"}
    deep_colors = {"UCB-HARE": "#0b5394", "Welfarist UCB": "#b45f06"}

    for key, regret_array in results_dict.items():
        try:
            algo, k_val_str = key.rsplit("_", 1)
            k_val = int(k_val_str[1:] if k_val_str.startswith("k") else k_val_str)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Expected result keys like 'UCB-HARE_k10'; received {key!r}"
            ) from exc
        if algo not in base_colors:
            raise ValueError(f"Unsupported algorithm in result key {key!r}")

        if k_val == 10:
            color, linestyle, linewidth, zorder = deep_colors[algo], "--", 2.0, 4
        else:
            color, linestyle, linewidth, zorder = base_colors[algo], "-", 2.5, 3

        regret = np.asarray(regret_array, dtype=float)
        if regret.ndim != 1:
            raise ValueError(f"Regret series {key!r} must be one-dimensional")
        ts = np.arange(1, regret.size + 1)

        # Begin at the requested 10^2 horizon and thin points for rendering.
        keep = np.flatnonzero(ts >= 1e2)[::100]
        if keep.size == 0 and ts.size:
            keep = np.array([ts.size - 1])
        ts_down, regret_down = ts[keep], regret[keep]
        # Logarithmic y axes cannot display zero or negative regret values.
        valid = np.isfinite(regret_down) & (regret_down > 0)
        ax.plot(
            ts_down[valid], regret_down[valid],
            label=rf"{algo} ($k={k_val}$)", color=color,
            linestyle=linestyle, linewidth=linewidth, zorder=zorder,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Round $t$ (log scale)", fontsize=14)
    ax.set_ylabel(rf"${p}$-mean regret (log scale)", fontsize=14)
    ax.set_title(rf"Fairness level ($p = {p}$)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower left", prop={"size": 12})
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved successfully to {RESULTS_DIR / filename}")
