import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
import pandas as pd


np.random.seed(42)

# careful, i had many type errors when building tmp_estimators, needs to check if the original code here (in __main__) still works
# 1. Age-dependent division rate
# 1.1. Lineage at divisions



def f_density(a, B):
    steps = np.arange(0, a, 0.01)
    integral = np.sum(B(steps)) * steps
    return B(a) * np.exp(-integral)


def _to_array(B, grid):
    """converts the function B into a numpy array because faster and simulation needs to be fast """
    return np.asarray(B(grid))


def sample_division_age(B, B_max):
    """Simulate one division using thinning method"""
    
    a=0.0 #we always start at age zero
    
    while True:
        a += np.random.exponential(1/B_max) #accumulation of arrival/jumping times
        if np.random.rand() <= B(a)/B_max: #probability of acceptation
            return a


# a_max and Xbar will come from real data
def simulate_lineage_age(B, grid, num_samples, growth_rate, Xbar):
    """Simulate many samples"""
    
    B_arr = _to_array(B, grid)
    B_max = np.max(B_arr)*1.1 #safety margin
    
    A = []
    Xb = []
    Xd = []
    X_current = Xbar #initialisation, but maybe it influences data too much?? --> burn in??

    for _ in range(num_samples): #tqdm(range(num_samples)):
        A_div = sample_division_age(B, B_max)
        X_div = X_current * np.exp(growth_rate*A_div)
        A.append(np.round(A_div,3))
        Xb.append(np.round(X_current, 3))
        Xd.append(np.round(X_div, 3))
        X_current = X_div / 2
    
    return np.column_stack((A, Xb, Xd))



def sample_division_size(x_birth, B, growth_rate):
    
    u = np.random.uniform(0, 1)
    
    target = -np.log(u)  

    def residual(x):
        integral, _ = quad(B, x_birth, x) # computes integral on [x_birth, x]
        return integral - target

    #produces upper bound on x, keeps multiply by 2 until cumB is greater than 0 
    x_upper = x_birth * 2
    for _ in range(10):
        if residual(x_upper) > 0:
            break
        x_upper *= 2 

    x_div = brentq(residual, x_birth, x_upper, xtol=1e-8) #numerical solver to find zero on [x_birth, x]
    return x_div


def simulate_lineage_size(Xbar, B, growth_rate, num_samples):
    A, Xb, Xd = [], [], []
    X_current = Xbar

    for _ in range(num_samples):
        X_div = sample_division_size(X_current, B, growth_rate)
        A_div = (1 / growth_rate) * np.log(X_div / X_current)

        A.append(np.round(A_div, 3))
        Xb.append(np.round(X_current, 3))
        Xd.append(np.round(X_div, 3))

        X_current = X_div / 2  

    return np.column_stack((A, Xb, Xd))




if __name__ == '__main__':

    import matplotlib.pyplot as plt
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
        growth_rate = 0.545
        grid = np.linspace(0, a_max, 2000)
        
        synthetic_data = simulate_lineage_age(B_power, grid, N, growth_rate, Xbar)
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
    
        synthetic_size_data = simulate_lineage_size(1, B_power, growth_rate, N)
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
        
