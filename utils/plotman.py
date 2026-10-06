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
                ts, lower, upper, color=color, alpha=0.18, linewidth=0,
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


def plot_mt(ts, mt_dict, filename="expt_C_mc.png", smoothed_data=None):
    original_ts = np.asarray(ts)
    ts, data = _downsample(original_ts, mt_dict)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    if smoothed_data is not None:
        _, smoothed = _downsample(original_ts, smoothed_data)
        _draw_bands(axes[0], ts, smoothed)
        axes[0].set_title("500-round moving average")
    _draw_bands(axes[1], ts, data)
    axes[1].set_title(r"Per-round estimate of $m_t$")
    for ax in axes:
        ax.set_xlabel('Round $t$')
        ax.legend(loc='best', frameon=True, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Expected arm mean $m_t=\mathbb{E}[\mu_{I_t}]$")
    fig.tight_layout()
    _save(fig, filename)
    plt.close(fig)


def plot_k_comparison(ts, curves, p, filename):
    ts = np.asarray(ts)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    mask = ts >= 1e2
    for name, value in curves.items():
        y = np.asarray(value)
        color = "blue" if name.startswith("UCB-HARE") else "orange"
        arm_count = name.rsplit("k=", 1)[-1].strip()
        linestyle = "--" if arm_count == "10" else "-"
        ax.plot(ts[mask], y[mask], color=color, linestyle=linestyle, linewidth=2, label=name)
    ax.set(
        xlabel="Round $t$ (log scale)", ylabel=r"$p$-mean regret",
        title=rf"Fairness level $p = {p:g}$",
    )
    ax.set_xscale("log")
    # Keep the compact in-axes placement used by the original regret plots.
    # The four entries fit in the upper-right without obscuring the early-time
    # comparison, and avoid reserving a large strip above the axes.
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    plt.close(fig)
