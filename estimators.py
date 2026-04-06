import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def kernel_estimation_fast(a_grid, observations, alpha):
    
    obs = np.asarray(observations)
    diff = a_grid[:, np.newaxis] - obs[np.newaxis, :]
    
    kernels = (1.0 / (np.sqrt(2 * np.pi) * alpha)) * np.exp(-0.5 * (diff / alpha)**2)
    numerator = np.sum(kernels, axis=1)
    denominator = np.sum(obs[np.newaxis, :] >= a_grid[:, np.newaxis], axis=1)
    
    # naive approach to tackle division by 0
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

def find_best_alpha(a_grid, observations, B_true_func, alphas):
    obs = np.asarray(observations)
    best_alpha = None
    min_mse = float('inf')
    mse_history = []
    
    b_true = B_true_func(a_grid)
    
    # Pre-calculate mask for data scarcity (at least 10 cells at risk)
    # This prevents the MSE from being ruined by the noisy "tail"
    risk_counts = np.sum(obs[np.newaxis, :] >= a_grid[:, np.newaxis], axis=1)
    mask = risk_counts > 10
    
    if not np.any(mask):
        return None, np.nan, []

    for alpha in alphas:
        b_hat = kernel_estimation_fast(a_grid, obs, alpha)
        
        # Calculate MSE only where data is sufficient
        mse = np.mean((b_hat[mask] - b_true[mask])**2)
        mse_history.append(mse)
        
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            
    return best_alpha, min_mse, mse_history



# test

PATH_SYNTH_AGE_LIN = "data/lin_synthetic_ages.txt"
# Assuming the file is space-separated based on your previous print output
synth_lin_age = pd.read_csv(PATH_SYNTH_AGE_LIN, sep=r'\s+', header=None, names=["ad"])
synth_real_ages = synth_lin_age["ad"]

def B_power(a):
    return a**1.1 


age_points = np.linspace(0, 4, 100)
alphas = np.linspace(0.05, 0.5, 50) 

best_a, min_m, history = find_best_alpha(age_points, synth_real_ages, B_power, alphas)

if best_a is not None:
    print(f"Best Alpha found: {best_a:.4f}")
    print(f"Minimum MSE: {min_m:.6f}")

    # Final Estimation
    B_estimated = kernel_estimation_fast(age_points, synth_real_ages, best_a)
    B_synthetic = B_power(age_points)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(age_points, B_synthetic, color='black', lw=2, label='Synthetic $B(a) = a^{1.1}$')
    plt.plot(age_points, B_estimated, color='blue', linestyle='--', lw=2, label=f'Estimated $\hat{{B}}$ ($\\alpha$={best_a:.2f})')
    
    plt.title('Division Rate Estimation: Synthetic vs. Kernel Estimator')
    plt.xlabel("Age at division")
    plt.ylabel("Division rate $B(a)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "outputs/synthetic_vs_estimated.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()
else:
    print("Optimization failed: Not enough data points 'at risk'.")