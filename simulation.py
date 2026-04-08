import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

np.random.seed(42)

# careful, i had many type errors when building tmp_estimators, needs to check if the original code here (in __main__) still works
# 1. Age-dependent division rate
# 1.1. Lineage at divisions



def f_density(a, B):
    steps = np.arange(0, a, 0.01)
    integral = np.sum(B(steps)) * steps
    return B(a) * np.exp(-integral)

def sample_division_age(B, a_max, B_max):
    """Simulate one division using thinning method"""
    a=0.0
    while True:
        a += np.random.exponential(1/B_max) #next jump
        if a > a_max * 1.1 : #safety margin
            return a_max
        idx = int(round(a / 0.01))
        idx = max(0, min(idx, len(B) - 1)) #TODO: maybe needs a better implementation to handle indices
        if np.random.rand() <= B[idx]/B_max: #probability of acceptation
            return a

# a_max and Xbar will come from real data
def simulate_lineage_age(B, num_samples, growth_rate, a_max, Xbar):
    """Simulate many samples"""
    grid = np.linspace(0, a_max, 1000)
    if type(B) == np.ndarray:
        B_max = np.max(B)*1.1
    else :
        B_max = np.max(B(grid)) * 1.1 #safety margin
    A = []
    Xb = []
    Xd = []
    X_current = Xbar

    for _ in range(num_samples): #tqdm(range(num_samples)):
        a_div = sample_division_age(B, a_max, B_max)
        x_div = X_current*np.exp(growth_rate*a_div)
        A.append(np.round(a_div,3))
        Xb.append(np.round(X_current, 3))
        Xd.append(np.round(x_div, 3))
        X_current = x_div / 2
    
    return np.column_stack((A, Xb, Xd))


if __name__ == '__main__':

    def B_power(a, power=2):
        return a**power
    N=2000
    power=2

    PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"
    lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])
    real_A = lin["ad"]
    real_Xb = lin["sb"]
    real_Xd = lin["sd"]
    a_max = np.max(real_A)
    Xbar = np.mean(real_Xb)
    print("real a_max:", a_max, "/ real X_bar:", Xbar)

    growth_rate = 0.53325 #found manually for B_power (power=2), to avoid explosion or zero sizes

    synthetic_data = simulate_lineage_age(B_power, N, growth_rate, a_max, Xbar)
    print(synthetic_data.shape)
    np.savetxt("data/synthetic_lin_age_model.txt", synthetic_data, delimiter=",")
    synthetic_A = synthetic_data[:,0]
    synthetic_Xb = synthetic_data[:,1]
    synthetic_Xd = synthetic_data[:,2]
    synthetic_A_max = np.round(np.max(synthetic_A),3)
    synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
    print("fake a_max:", synthetic_A_max, "/ fake Xbar:", synthetic_Xbar)

    #Visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 6))
    #real data
    axes[0,0].hist(real_A, bins=40, density=True)
    axes[0,0].set_title("Real ages distribution")
    axes[0,0].set_xlabel("Age at division")
    axes[0,0].set_ylabel("Density")
    axes[0,0].grid()

    axes[1,0].hist(real_Xb, bins=40, density=True)
    axes[1,0].set_title("Real sizes at birth distribution")
    axes[1,0].set_xlabel("Size at birth")
    axes[1,0].set_ylabel("Density")
    axes[1,0].grid()

    axes[2,0].hist(real_Xd, bins=40, density=True)
    axes[2,0].set_title("Real sizes at division distribution")
    axes[2,0].set_xlabel("Sizes at division")
    axes[2,0].set_ylabel("Density")
    axes[2,0].grid()

    # synthetic data
    axes[0,1].hist(synthetic_A, bins=40, density=True)
    axes[0,1].set_title(f"Synthetic ages distribution for B(a)=a**{power}")
    axes[0,1].set_xlabel("Age at division")
    axes[0,1].grid()

    axes[1,1].hist(synthetic_Xb, bins=40, density=True)
    axes[1,1].set_title(f"Synthetic sizes at birth distribution for B(a)=a**{power}")
    axes[1,1].set_xlabel("Size at birth")
    axes[1,1].grid()

    axes[2,1].hist(synthetic_Xd, bins=40, density=True)
    axes[2,1].set_title(f"Synthetic sizes at division distribution for B(a)=a**{power}")
    axes[2,1].set_xlabel("Size at division")
    axes[2,1].grid()

    plt.tight_layout()
    plt.savefig('outputs/synthetic_lin_age_model.png')


