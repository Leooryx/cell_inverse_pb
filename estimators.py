import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

np.random.seed(42)

def gaussian_kernel(z, alpha):
    return (1.0 / (np.sqrt(2 * np.pi) * alpha)) * np.exp(-0.5 * (z / alpha)**2)

def kernel_estimation_cdf(observations, alpha):
    obs = np.asarray(observations)
    a_grid = np.linspace(np.min(obs), np.max(obs), len(obs))
    
    numer_list = []
    denom_list = []
    
    for a in a_grid:
        z = (a - observations) / alpha 
        numer = np.sum(gaussian_kernel(a - obs, alpha))
        denom = np.sum(1-norm.cdf(z))
        numer_list.append(numer)
        denom_list.append(denom)
    
    B_hat = np.array(numer_list) / np.array(denom_list) 
    
    return B_hat



def find_best_alpha(observations, B, alphas):
    obs = np.asarray(observations)
    a_grid = np.linspace(min(obs), max(obs), len(obs))
    best_alpha = None
    min_mse = float('inf')
    mse_history = []
    
    B_true = B(a_grid)

    for alpha in tqdm(alphas):
        B_hat = kernel_estimation_cdf(obs, alpha)
        valid = ~np.isnan(B_hat)
        mse = np.mean((B_hat[valid] - B_true[valid])**2)
        mse_history.append(mse)
        
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            
    return best_alpha, min_mse, mse_history


if __name__ == '__main__':

    # test
    PATH_SYNTH_AGE_LIN = "data/synthetic_lin_age_model.txt"
    synth_lin_age = pd.read_csv(PATH_SYNTH_AGE_LIN, header=None, names=["ad", "sb", "sd"])
    synth_real_ages = synth_lin_age["ad"]

    def B_power(a):
        return a**2


    age_points = np.linspace(min(synth_real_ages), max(synth_real_ages), len(synth_real_ages))
    alphas = np.linspace(0.05, 0.5, 50) 

    best_a, min_m, history = find_best_alpha(synth_real_ages, B_power, alphas)

    print(f"Best Alpha found: {best_a:.4f}")
    print(f"Minimum MSE: {min_m:.3f}")

    B_estimated = kernel_estimation_cdf(synth_real_ages, best_a)
    B_synthetic = B_power(age_points)


    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: MSE history ---
    axes[0].plot(alphas, history, lw=2)
    axes[0].axvline(best_a, linestyle='--', label=f'Best alpha = {best_a:.3f}')
    axes[0].set_title("MSE vs Alpha")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # --- Right: B vs B_hat ---
    axes[1].plot(age_points, B_synthetic, color='black', lw=2, label='Synthetic B')
    axes[1].plot(age_points, B_estimated, color='blue', linestyle='--', lw=2,
                label=f'Estimated B, alpha={best_a:.2f}, MSE={min_m:.3f}')
    axes[1].set_title('Division Rate Estimation')
    axes[1].set_xlabel("Age at division")
    axes[1].set_ylabel("Division rate B(a)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Layout
    plt.tight_layout()

    output_path = "outputs/synthetic_vs_estimated.png"
    # with B_power=2, we have huge variance at large ages, not surprising because we have very few samples of old ages given this form of B
    plt.savefig(output_path)
    plt.close()



