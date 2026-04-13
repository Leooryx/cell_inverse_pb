import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from plots import plot_estimator_results

np.random.seed(42)

def gaussian_kernel(z, alpha):
    return (1.0 / (np.sqrt(2 * np.pi) * alpha)) * np.exp(-0.5 * (z / alpha)**2)

# estimator for lineage age 
def B_lineage_age(observations, alpha):
    obs = np.asarray(observations)
    n=len(obs)
    a_grid = np.linspace(np.min(obs), np.max(obs), n)
    
    numer_list = []
    denom_list = []

    omega_n = np.sqrt(n)
    
    for a in a_grid:
        z = (a - observations) / alpha 
        numer = np.sum(gaussian_kernel(a - obs, alpha))
        #denom = max( np.sum(obs >= a), omega_n)
        denom = np.sum(1-norm.cdf(z)) #with cdf so its never equal to zero
        numer_list.append(numer)
        denom_list.append(denom)
    
    B_hat = np.array(numer_list) / np.array(denom_list) 
    
    return B_hat


def find_best_alpha(estimator, observations, B, alphas):
    obs = np.asarray(observations)
    a_grid = np.linspace(min(obs), max(obs), len(obs))
    best_alpha = None
    min_mse = float('inf')
    mse_history = []
    
    B_true = B(a_grid)

    for alpha in tqdm(alphas):
        B_hat = estimator(obs, alpha)
        valid = ~np.isnan(B_hat)
        mse = np.mean((B_hat[valid] - B_true[valid])**2)
        mse_history.append(mse)
        
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            
    return best_alpha, min_mse, mse_history

def verify_estimator(estimator, truth, data_from_truth):
    points = np.linspace(min(data_from_truth), max(data_from_truth), len(data_from_truth))
    alphas = np.linspace(0.01, 1, 100) 
    best_alpha, min_m, history = find_best_alpha(estimator, data_from_truth, truth,  alphas)
    print(f"Best Alpha found: {best_alpha:.4f}")
    print(f"Minimum MSE: {min_m:.3f}")
    estimation = B_lineage_age(data_from_truth, best_alpha)
    truth_profile = B_power(points)

    output_path="synthetic_vs_estimated.png"
    plot_estimator_results(alphas, history, best_alpha, points, truth_profile, estimation, min_m, output_path, "age")



# test to verify if the estimator can recover the division rate if it was known!
if __name__ == '__main__':
    from plots import plot_estimator_results

    PATH_SYNTH_AGE_LIN = "data/synthetic_lin_age_model.txt"
    synth_lin_age = pd.read_csv(PATH_SYNTH_AGE_LIN, header=None, names=["ad", "sb", "sd"])
    synth_real_ages = synth_lin_age["ad"]

    def B_power(a):
        return a**1
    
    verify_estimator(B_lineage_age, B_power, synth_real_ages)


    

    

    

    

   