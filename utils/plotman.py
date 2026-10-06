"""Publication plots in the paper's existing blue/orange/green palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"UCB-HARE": "blue", "Welfarist UCB": "orange", "Explore-Then-UCB": "green"}


def _downsample(ts, arrays, num_points=10000):
    if len(ts) <= num_points:
        return np.asarray(ts), arrays
    idx = np.linspace(0, len(ts) - 1, num_points, dtype=int)
    return np.asarray(ts)[idx], {key: tuple(np.asarray(part)[idx] for part in value) for key, value in arrays.items()}


def _band(value):
    if isinstance(value, tuple) and len(value) == 3:
        return value
    y = np.asarray(value)
    return y, y, y


def _draw_bands(ax, ts, data, labels=None):
    for name, value in data.items():
        y, lower, upper = _band(value)
        color = COLORS.get(name, None)
        label = labels.get(name, name) if labels else name
        ax.plot(ts, y, label=label, linewidth=2.2 if name == "UCB-HARE" else 1.9, color=color, zorder=3 if name == "UCB-HARE" else 2)
        if np.any(np.asarray(upper) > np.asarray(lower)):
            ax.fill_between(ts, lower, upper, color=color, alpha=0.18, linewidth=0, zorder=1)


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
    fig.savefig(f"results/{filename}", dpi=300)
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
    fig.savefig(f"results/{filename}", dpi=300)
    plt.close(fig)


def plot_conditional_policy_value(ts, conditional_values, filename):
    ts, data = _downsample(ts, {"UCB-HARE": (np.asarray(conditional_values),)})
    y = data["UCB-HARE"][0]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(ts, y, color="blue", linewidth=1.2, label="One-run $\\tilde m_t$")
    ax.set(xlabel='Round $t$', ylabel=r"Conditional expected reward $\tilde m_t$")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{filename}", dpi=300)
    plt.close(fig)


def plot_mt(ts, mt_dict, filename="expt_C_mc.png"):
    ts, data = _downsample(ts, mt_dict)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _draw_bands(ax, ts, data)
    ax.set(xlabel='Round $t$', ylabel=r"Estimated ex-ante reward $m_t$")
    ax.legend(loc='best', frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{filename}", dpi=300)
    plt.close(fig)


def plot_k_comparison(ts, curves, p, filename):
    ts = np.asarray(ts)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    mask = ts >= 1e2
    for name, value in curves.items():
        y = np.asarray(value)
        color = "blue" if name.startswith("UCB-HARE") else "orange"
        linestyle = "-" if "k=10" in name else "--"
        ax.plot(ts[mask], y[mask], color=color, linestyle=linestyle, linewidth=2, label=name)
    ax.set(xlabel="Round $t$ (log scale)", ylabel=r"$p$-mean regret")
    ax.set_xscale("log")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), frameon=True, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"results/{filename}", dpi=300)
    plt.close(fig)
