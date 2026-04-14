import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from simulation import simulate_lineage_age
from estimators import B_lineage_age


# we want to optimize the kernel regression with cdf to minimize wasserstein distance.abs

def grid_search_alpha(observations, alpha_grid, simulator, growth_rate, a_max, Xbar):
    obs = np.asarray(observations)
    results=[]
    best_alpha=None
    min_dist = float('inf')
    for alpha in tqdm(alpha_grid):
        B_hat = B_lineage_age(obs, alpha)
        simulated_data = simulator(Xbar, B_hat, growth_rate, len(obs))[:,0]
        dist = wasserstein_distance(obs, simulated_data)
        results.append(dist)

        if dist < min_dist:
            min_dist = dist
            best_alpha = alpha
            Best_B_hat = B_hat
    print("minimum distance:", min_dist)
        
    return best_alpha, np.asarray(results), Best_B_hat


N=2000

PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"
lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])
real_A = lin["ad"]
real_Xb = lin["sb"]
real_Xd = lin["sd"]
a_max = np.max(real_A)
Xbar = np.mean(real_Xb)
#print("real a_max:", a_max, "/ real X_bar:", Xbar)

growth_rate = 0.032 #according to regression

alpha_grid = np.linspace(0.01, 1, 10)


best_alpha, dist_hist, Best_B_hat = grid_search_alpha(real_A, alpha_grid, simulate_lineage_age, growth_rate, a_max, Xbar)

print("best alpha:", best_alpha)
print("min dist:", np.min(dist_hist))

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Wasserstein history ---
axes[0].plot(alpha_grid, dist_hist, lw=2)
axes[0].axvline(best_alpha, linestyle='--', label=f'Best alpha = {best_alpha:.3f}')
axes[0].set_title("Wasserstein distance vs Alpha")
axes[0].set_xlabel("Alpha")
axes[0].set_ylabel("Wasserstein distance")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# --- Right: B vs B_hat ---
age_points = np.linspace(min(real_A), max(real_A), len(real_A))
axes[1].plot(age_points, Best_B_hat, color='blue', linestyle='--', lw=2,
             label=f'Estimated B, alpha={best_alpha:.2f}, Min_Wasserstein={np.min(dist_hist):.3f}')
axes[1].set_title('Division Rate Estimation')
axes[1].set_xlabel("Age at division")
axes[1].set_ylabel("Division rate B(a)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Layout
plt.tight_layout()

output_path = "outputs/7_kernel_lineage_age.png"
plt.savefig(output_path)

# with B_power=2, we have huge variance at large ages, not surprising because we have very few samples of old ages given this form of B

