import numpy as np
import matplotlib.pyplot as plt

_STYLE = dict(bins=40, density=True)


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(f"outputs/{path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# simulation.py plots
# ---------------------------------------------------------------------------

def plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path):
    """ real data (left) vs synthetic data (right) distributions."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    # --- Real data (left column) ---
    _hist(axes[0, 0], real_A, "Real ages distribution", "Age at division")
    _hist(axes[1, 0], real_Xb, "Real sizes at birth distribution", "Size at birth")
    _hist(axes[2, 0], real_Xd, "Real sizes at division distribution", "Size at division")

    # --- Synthetic data (right column) ---
    _hist(axes[0, 1], synthetic_A,   "Synthetic ages","Age at division",  ylabel=False)
    _hist(axes[1, 1], synthetic_Xb,  "Synthetic sizes at birth", "Size at birth",   ylabel=False)
    _hist(axes[2, 1], synthetic_Xd,  "Synthetic sizes at division", "Size at division", ylabel=False)

    _save(fig, output_path)


def _hist(ax, data, title, xlabel, ylabel=True):
    ax.hist(data, **_STYLE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Density")
    ax.grid(alpha=0.4)


# ---------------------------------------------------------------------------
# estimator.py plots
# ---------------------------------------------------------------------------

def plot_estimator_results(alphas, mse_history, best_alpha, points, B_synthetic, B_estimated, min_mse, output_path, age_or_size):
    """MSE vs alpha (left) + B vs B_hat (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE curve
    axes[0].plot(alphas, mse_history, lw=2)
    axes[0].axvline(best_alpha, linestyle="--", label=f"Best alpha = {best_alpha:.3f}")
    axes[0].set_title("MSE vs Alpha")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # B vs B_hat
    axes[1].plot(points, B_synthetic, color="black", lw=2, label="Synthetic B")
    axes[1].plot(points, B_estimated, color="blue", lw=2, ls="--", label=f"Estimated B, alpha={best_alpha:.3f},  MSE={min_mse:.3f}")
    axes[1].set_title("Division rate estimation")
    axes[1].set_xlabel(f"{age_or_size} at division")
    axes[1].set_ylabel("Division rate B(a)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# main.py plots
# ---------------------------------------------------------------------------

def plot_main_results(alpha_grid, dist_hist, best_alpha, real_data, Best_B_hat, output_path, age_or_size):
    """Wasserstein vs alpha (left) + best B_hat (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    min_dist   = np.min(dist_hist)
    points = np.linspace(np.min(real_data), np.max(real_data), len(real_data))

    # Wasserstein curve
    axes[0].plot(alpha_grid, dist_hist, lw=2)
    axes[0].axvline(best_alpha, linestyle="--", label=f"Best alpha = {best_alpha:.3f}")
    axes[0].set_title("Wasserstein distance vs alpha")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Wasserstein distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Best B_hat
    axes[1].plot(points, Best_B_hat(points), color="blue", lw=2, ls="--",
                 label=f"Estimated B  alpha={best_alpha:.2f}  W={min_dist:.3f}")
    axes[1].set_title("Best Division Rate Estimate")
    axes[1].set_xlabel(f"{age_or_size} at division")
    axes[1].set_ylabel("Division rate B")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    _save(fig, output_path)