import numpy as np
from scipy.stats import wasserstein_distance 
from tqdm import tqdm
import pandas as pd

from plots import plot_main_results, plot_simulation_comparison
from simulation import simulate_lineage_age, simulate_lineage_size
from estimators import B_lineage_age, B_lineage_size


# we want to optimize the kernel regression with cdf to minimize wasserstein distance.abs

def grid_search_alpha(observations, estimator, alpha_grid, simulator, growth_rate, v_max, Xbar):
    obs = np.asarray(observations)
    results=[]
    best_alpha=None
    min_dist = float('inf')
    for alpha in tqdm(alpha_grid):
        np.random.seed(42) #same randomness for all samples
        B_hat = estimator(obs, alpha)[1] #[1] for function instead of np.array
        simulated_data = simulator(Xbar=Xbar, v_max=v_max, B_func=B_hat, growth_rate=growth_rate, num_samples=len(obs))[:,0]
        dist = wasserstein_distance(obs, simulated_data) #/ np.std(obs)
        results.append(dist)

        if dist < min_dist:
            min_dist = dist
            best_alpha = alpha
            Best_B_hat = B_hat
    print("minimum distance:", min_dist)
    print("best alpha:", best_alpha)
        
    return best_alpha, np.asarray(results), Best_B_hat


if __name__ == "__main__":

    test_age = True
    test_size = True

    PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"
    lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])
    real_A = lin["ad"]
    scale_age = np.std(real_A)
    real_A_normalized = real_A / scale_age
    real_Xb = lin["sb"]
    scale_size = np.std(real_Xb)
    real_Xb_normalized = real_Xb / scale_size
    real_Xd = lin["sd"]
    a_max = np.max(real_A) / scale_age
    Xbar = np.mean(real_Xb) / scale_size
    x_max = np.max(real_Xd) / scale_size

    growth_rate = 0.032 #according to regression
    alpha_grid = np.linspace(0.01, 20, 100)


    

    if test_age:
        #  (we can add the other plots to compare all the distributions!!)
        best_alpha, dist_hist, Best_B_hat_normalized = grid_search_alpha(real_A_normalized, B_lineage_age, alpha_grid, simulate_lineage_age, growth_rate, a_max, Xbar)
        output_path = "7_kernel_lineage_age.png"
        Best_B_hat = lambda a: Best_B_hat_normalized(a / scale_age) / scale_age #un-normalized division rate
        plot_main_results(alpha_grid, dist_hist, best_alpha, real_A, Best_B_hat, output_path, "age")

        synthetic_data = simulate_lineage_age(Xbar*scale_age, a_max*scale_age, Best_B_hat, growth_rate, 2000)
        synthetic_A = synthetic_data[:,0]
        synthetic_Xb = synthetic_data[:,1]
        synthetic_Xd = synthetic_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max*scale_age, "/ real X_bar:", Xbar*scale_age)
        print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)


        output_path = "7_synthetic_lin_age_model.png"

        plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path)
    
    if test_size:
        best_alpha, dist_hist, Best_B_hat_normalized = grid_search_alpha(real_Xb_normalized, B_lineage_size, alpha_grid, simulate_lineage_size, growth_rate, x_max, Xbar)
        output_path = "7_kernel_lineage_size.png"
        Best_B_hat = lambda x: Best_B_hat_normalized(x / scale_size) / scale_size #un-normalized division rate
        plot_main_results(alpha_grid, dist_hist, best_alpha, real_Xb, Best_B_hat, output_path, "size")
        
        synthetic_size_data = simulate_lineage_size(Xbar*scale_size, Best_B_hat, growth_rate, 2000, x_max*scale_size, burn_in=200)
        synthetic_A = synthetic_size_data[:,0]
        synthetic_Xb = synthetic_size_data[:,1]
        synthetic_Xd = synthetic_size_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max*scale_age, "/ real X_bar:", Xbar*scale_size)
        print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)
        
        output_path = 'synthetic_lin_size_model.png'
        plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path)
