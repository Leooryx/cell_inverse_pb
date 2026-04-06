import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaussian_kernel(z, alpha):
    return (1.0 / (np.sqrt(2 * np.pi) * alpha)) * np.exp(-0.5 * (z / alpha)**2)

def kernel_estimation(observations, alpha):
    obs = np.asarray(observations)
    a_grid = np.linspace(np.min(obs), np.max(obs), len(obs))
    
    B_hat = []
    denom_list = []
    
    for a in a_grid:
        numerator = np.sum(gaussian_kernel(a - obs, alpha))
        denominator = np.sum(obs >= a)
        
        B_hat.append(numerator)
        denom_list.append(denominator)
    
    B_hat = np.array(B_hat)
    denom = np.array(denom_list)
    
    # handling division by zero by setting a mask to compute survival function only when there are more than k individuals
    k = 1
    valid = denom > k
    
    result = np.full_like(B_hat, np.nan)
    result[valid] = B_hat[valid] / denom[valid]
    
    return result

def find_best_alpha(observations, B, alphas):
    obs = np.asarray(observations)
    a_grid = np.linspace(min(obs), max(obs), len(obs))
    best_alpha = None
    min_mse = float('inf')
    mse_history = []
    
    B_true = B(a_grid)

    for alpha in alphas:
        B_hat = kernel_estimation(obs, alpha)
        valid = ~np.isnan(B_hat)
        # Calculate MSE only where data is sufficient
        mse = np.mean((B_hat[valid] - B_true[valid])**2)
        mse_history.append(mse)
        
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            
    return best_alpha, min_mse, mse_history



# test

PATH_SYNTH_AGE_LIN = "data/lin_synthetic_ages.txt"
synth_lin_age = pd.read_csv(PATH_SYNTH_AGE_LIN, header=None, names=["ad"])
synth_real_ages = synth_lin_age["ad"]
print(synth_real_ages)

def B_power(a):
    return a**1.1 


age_points = np.linspace(min(synth_real_ages), max(synth_real_ages), len(synth_real_ages))
alphas = np.linspace(0.05, 0.5, 50) 

best_a, min_m, history = find_best_alpha(synth_real_ages, B_power, alphas)

print(f"Best Alpha found: {best_a:.4f}")
print(f"Minimum MSE: {min_m:.6f}")

B_estimated = kernel_estimation(synth_real_ages, best_a)
B_synthetic = B_power(age_points)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(age_points, B_synthetic, color='black', lw=2, label='Synthetic B')
plt.plot(age_points, B_estimated, color='blue', linestyle='--', lw=2, label=f'Estimated B, alpha={best_a:.2f}')

plt.title('Division Rate Estimation: Synthetic vs. Kernel Estimator')
plt.xlabel("Age at division")
plt.ylabel("Division rate $B(a)$")
plt.legend()
plt.grid(True, alpha=0.3)

output_path = "outputs/synthetic_vs_estimated.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
plt.show()
