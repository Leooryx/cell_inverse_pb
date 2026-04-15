import numpy as np
from scipy.stats import wasserstein_distance 
from tqdm import tqdm
import pandas as pd

from plots import plot_main_results, plot_simulation_comparison
from simulation import simulate_lineage_age, simulate_lineage_size
from estimators import B_lineage_age, B_lineage_size


# we want to optimize the kernel regression with cdf to minimize wasserstein distance.abs

def grid_search_alpha(observations, estimator, alpha_grid, simulator, growth_rate, v_max, Xbar, age_or_size):
    obs = np.asarray(observations)
    results=[]
    best_alpha=None
    min_dist = float('inf')
    for alpha in tqdm(alpha_grid):
        np.random.seed(42) #same randomness for all samples
        B_hat = estimator(obs, alpha)[1] #[1] for function instead of np.array
        index = 0 if age_or_size=="age" else 2
        simulated_data = simulator(Xbar=Xbar, v_max=v_max, B_func=B_hat, growth_rate=growth_rate, num_samples=len(obs), burn_in=0)[:,index]
        dist = wasserstein_distance(obs, simulated_data) 
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
    real_Xb = lin["sb"]
    real_Xd = lin["sd"]
    a_max = np.max(real_A)
    Xbar = np.mean(real_Xb)
    x_max = np.max(real_Xd) 

    growth_rate = 0.032 #according to regression
    alpha_grid = np.linspace(0.01, 2, 101)



    if test_age:
        #  (we can add the other plots to compare all the distributions!!)
        best_alpha, dist_hist, Best_B_hat = grid_search_alpha(real_A, B_lineage_age, alpha_grid, simulate_lineage_age, growth_rate, a_max, Xbar, "age")
        output_path = "7_kernel_lineage_age.png"
        plot_main_results(alpha_grid, dist_hist, best_alpha, real_A, Best_B_hat, output_path, "age")
        
        np.random.seed(42)
        synthetic_data = simulate_lineage_age(Xbar, a_max, Best_B_hat, growth_rate, 2000)
        synthetic_A = synthetic_data[:,0]
        synthetic_Xb = synthetic_data[:,1]
        synthetic_Xd = synthetic_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max, "/ real X_bar:", Xbar)
        print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)

        dist_age = np.round(wasserstein_distance(real_A, synthetic_A), 3)
        dist_sb = np.round(wasserstein_distance(real_Xb, synthetic_Xb), 3)
        dist_sd = np.round(wasserstein_distance(real_Xd, synthetic_Xd), 3)
        print("Summary: distance wrt Age:", dist_age)
        print("distance wrt Birth Size:", dist_sb) 
        print("distance wrt Division Size:", dist_sd)

        output_path = "7_synthetic_lin_age_model.png"
        plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path)
    

    if test_size:
        best_alpha, dist_hist, Best_B_hat_normalized = grid_search_alpha(real_Xb, B_lineage_size, alpha_grid, simulate_lineage_size, growth_rate, x_max, Xbar, "size")
        output_path = "7_kernel_lineage_size.png"
        plot_main_results(alpha_grid, dist_hist, best_alpha, real_Xb, Best_B_hat, output_path, "size")
        
        np.random.seed(42)
        synthetic_size_data = simulate_lineage_size(Xbar, Best_B_hat, growth_rate, 2000, x_max, burn_in=200)
        synthetic_A = synthetic_size_data[:,0]
        synthetic_Xb = synthetic_size_data[:,1]
        synthetic_Xd = synthetic_size_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max, "/ real X_bar:", Xbar)
        print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)

        dist_age = np.round(wasserstein_distance(real_A, synthetic_A), 3)
        dist_sb = np.round(wasserstein_distance(real_Xb, synthetic_Xb), 3)
        dist_sd = np.round(wasserstein_distance(real_Xd, synthetic_Xd), 3)
        print("Summary: distance wrt Age:", dist_age)
        print("distance wrt Birth Size:", dist_sb) 
        print("distance wrt Division Size:", dist_sd)
                
        output_path = '7_synthetic_lin_size_model.png'
        plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path)
