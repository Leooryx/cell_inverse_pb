import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import interp1d

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

    #omega_n = np.sqrt(n)
    
    for a in a_grid:
        z = (a - observations) / alpha 
        numer = np.sum(gaussian_kernel(a - obs, alpha))
        denom = np.sum(1-norm.cdf(z)) #with cdf so its never equal to zero
        numer_list.append(numer)
        denom_list.append(denom)
    
    B_hat_array = np.array(numer_list) / np.array(denom_list) 
    B_hat_func = interp1d(a_grid, B_hat_array, kind='linear', bounds_error=False, fill_value=(B_hat_array[0], B_hat_array[-1]))
    
    return [B_hat_array, B_hat_func]

def B_lineage_size(observations, alpha):
    obs = np.asarray(observations)
    x_grid = np.linspace(np.min(obs), np.max(obs), len(obs))

    numer_list = []
    denom_list = []

    for x in x_grid:
        z1 = (x - obs)/alpha
        z2 = (x/2 - obs)/alpha
        numer = np.sum(gaussian_kernel(x/2-obs, alpha))
        denom = np.sum(norm.cdf(z1) - norm.cdf(z2))
        numer_list.append(numer)
        denom_list.append(denom)

    B_hat_array = 0.5*np.array(numer_list) / np.array(denom_list)
    B_hat_func = interp1d(x_grid, B_hat_array, kind='linear', bounds_error=False, fill_value=(B_hat_array[0], B_hat_array[-1]))
    
    return [B_hat_array, B_hat_func]



def find_best_alpha(estimator, observations, B, alphas):
    obs = np.asarray(observations)
    a_grid = np.linspace(min(obs), max(obs), len(obs))
    best_alpha = None
    min_mse = float('inf')
    mse_history = []
    
    B_true = B(a_grid)

    for alpha in tqdm(alphas):
        B_hat = estimator(obs, alpha)[0]
        mse = np.mean((B_hat - B_true)**2)
        mse_history.append(mse)
        
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            
    return best_alpha, min_mse, mse_history



def verify_estimator(estimator, truth, data_from_truth, age_or_size):
    points = np.linspace(min(data_from_truth), max(data_from_truth), len(data_from_truth))
    alphas = np.linspace(0.01, 1, 100) 
    best_alpha, min_m, history = find_best_alpha(estimator, data_from_truth, truth,  alphas)
    print(f"Best Alpha found: {best_alpha:.4f}")
    print(f"Minimum MSE: {min_m:.3f}")
    estimation = estimator(data_from_truth, best_alpha)[0]
    truth_profile = B_power(points)

    output_path="synthetic_vs_estimated.png"
    plot_estimator_results(alphas, history, best_alpha, points, truth_profile, estimation, min_m, output_path, age_or_size)



# test to verify if the estimator can recover the division rate if it was known!
if __name__ == '__main__':
    test_age = True
    test_size = True
    
    def B_power(a):
        return a**1

    if test_age:
        PATH_SYNTH_AGE_LIN = "data/synthetic_lin_age_model.txt"
        synth_lin_age = pd.read_csv(PATH_SYNTH_AGE_LIN, header=None, names=["ad", "sb", "sd"])
        synth_real_ages = synth_lin_age["ad"]
        
        verify_estimator(B_lineage_age, B_power, synth_real_ages, "age")
    
    if test_size:
        PATH_SYNTH_SIZE_LIN = "data/synthetic_lin_size_model.txt"
        synth_lin_size = pd.read_csv(PATH_SYNTH_SIZE_LIN, header=None, names=["ad", "sb", "sd"])
        synth_real_sizes = synth_lin_size["sb"]
        
        verify_estimator(B_lineage_size, B_power, synth_real_sizes, "size")


    

    

    

    

   