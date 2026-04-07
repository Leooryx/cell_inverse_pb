import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# We use the rejection algorithm


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
        if np.random.rand() <= B(a)/B_max: #probability of acceptation
            return a

# a_max and Xbar will come from real data
def simulate_lineage(B, num_samples, growth_rate, a_max, Xbar):
    """Simulate many samples"""
    grid = np.linspace(0, a_max, 1000)
    B_max = np.max(B(grid)) * 1.1 #safety margin
    A = []
    Xb = []
    Xd = []
    X_current = Xbar

    for _ in tqdm(range(num_samples)):
        a_div = sample_division_age(B, a_max, B_max)
        x_div = X_current*np.exp(growth_rate*a_div)
        A.append(a_div)
        Xb.append(X_current)
        Xd.append(x_div)
        X_current = x_div / 2
    
    return [np.array(A), np.array(Xb), np.array(Xd)].T




def B_power(a, power=2):
    return a**power
N=2000
power=2


PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"
lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])
real_ages = lin["ad"]
real_Xd = lin["sd"]
a_max = np.max(real_ages)
Xbar = np.mean(real_Xd)
growth_rate = 0.78 #inspired from linear regression
synthetic_data = simulate_lineage(B_power, N, growth_rate, a_max, Xbar)
np.savetxt("data/lin_synthetic_ages.txt", synthetic_data)
synthetic_ages = synthetic_data[0]

#Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#real data
ax1.hist(real_ages, bins=40, density=True)
ax1.set_title("Real ages distribution")
ax1.set_xlabel("Age at division")
ax1.set_ylabel("Density")

# synthetic data
ax2.hist(synthetic_ages, bins=40, density=True)
ax2.set_title(f"Synthetic ages distribution for B(a)=a**{power}")
ax2.set_xlabel("Age at division")

plt.tight_layout()
plt.savefig('outputs/synthetic_ages.png')