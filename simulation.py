import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from tqdm import tqdm

# attention, functions in age and size do not have the same order of parameters

np.random.seed(42)

def _to_array(B, grid):
    """converts the function B into a numpy array because faster and simulation needs to be fast """
    return np.asarray(B(grid))


def sample_age_division(B_func, a_max, num_samples):
    grid = np.linspace(0, a_max, num_samples)
    B_vals = B_func(grid)
    integral = cumulative_trapezoid(B_vals, grid, initial=0)
    inv = interp1d(integral, grid, kind='linear', bounds_error=False, fill_value=(grid[0],grid[-1]))

    def sampler(n):
        U = np.random.uniform(0,1, size=n)
        targets = -np.log(U)
        return inv(targets)
    
    return sampler

def simulate_lineage_age(Xbar, v_max, B_func, growth_rate, num_samples, burn_in=200):
    sampler = sample_age_division(B_func, v_max, num_samples)
    all_ages = sampler(num_samples + burn_in)
    X_current = Xbar
    A, Xb, Xd = [], [], []
    for i, A_div in enumerate(all_ages):
        X_div = X_current * np.exp(growth_rate * A_div)
        if i >= burn_in:
            A.append(np.round(A_div, 3))
            Xb.append(np.round(X_current, 3))
            Xd.append(np.round(X_div, 3))
        X_current = X_div / 2
    
    return np.column_stack((A, Xb, Xd))


def sample_size_division(B_func, x_max, num_samples):

    grid = np.linspace(0, x_max, num_samples)
    B_vals = B_func(grid)
    H_vals = cumulative_trapezoid(B_vals, grid, initial=0)
    H_interp = interp1d(grid, H_vals, kind='linear',bounds_error=False,fill_value=(grid[0], grid[-1]))
    H_inv = interp1d(H_vals, grid, kind='linear',bounds_error=False,fill_value=(H_vals[0], H_vals[-1]))
    
    def sampler(x_birth):
        U = np.random.uniform(0, 1)
        target = H_interp(x_birth) - np.log(U)
        return float(H_inv(target))

    return sampler

def simulate_lineage_size(Xbar, B_func, growth_rate, num_samples, v_max, burn_in=200):

    sampler = sample_size_division(B_func, v_max, num_samples)

    X_current = Xbar
    A, Xb, Xd = [], [], []

    for i in range(num_samples + burn_in):
        X_div = sampler(X_current)
        A_div = (1 / growth_rate) * np.log(X_div / X_current)

        if i >= burn_in:
            A.append(np.round(A_div, 3))
            Xb.append(np.round(X_current, 3))
            Xd.append(np.round(X_div, 3))

        X_current = X_div / 2

    return np.column_stack((A, Xb, Xd))




if __name__ == '__main__':

    from plots import plot_simulation_comparison

    test_age = False
    test_size = True

    
    N=2000
    power=1

    PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"
    lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])
    real_A = lin["ad"]
    real_Xb = lin["sb"]
    real_Xd = lin["sd"]
    Xbar = np.mean(real_Xb)
    a_max=np.max(real_A)
    

    def B_power(a):
            return a**power

    if test_age:
        growth_rate = 0.5499
        grid = np.linspace(0, a_max, 2000)
        synthetic_data = simulate_lineage_age(1, a_max, B_power, growth_rate, N)
        np.savetxt("data/synthetic_lin_age_model.txt", synthetic_data, delimiter=",")
        synthetic_A = synthetic_data[:,0]
        synthetic_Xb = synthetic_data[:,1]
        synthetic_Xd = synthetic_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max, "/ real X_bar:", Xbar)
        print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)


        output_path = "synthetic_lin_age_model.png"

        plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path)



    if test_size:
        growth_rate = 0.6
        x_max = np.max(real_Xd)
        
        synthetic_size_data = simulate_lineage_size(Xbar, B_power, growth_rate, N, x_max, burn_in=200)
        np.savetxt("data/synthetic_lin_size_model.txt", synthetic_size_data, delimiter=",")
        
        synthetic_A = synthetic_size_data[:,0]
        synthetic_Xb = synthetic_size_data[:,1]
        synthetic_Xd = synthetic_size_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max, "/ real X_bar:", Xbar)
        print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)
        
        output_path = 'synthetic_lin_size_model.png'
        plot_simulation_comparison(real_A, real_Xb, real_Xd, synthetic_A, synthetic_Xb, synthetic_Xd, output_path)
        
