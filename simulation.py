import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# We use the rejection algorithm


# 1. Age-dependent division rate
# 1.1. Lineage at divisions

def simulate_lineage(B, n_samples, a_max=45.0):
    samples = []
    
    #get max of f
    x_test = np.linspace(0.01, a_max, 200)
    def f_density(a):
        steps = np.arange(0, a, 0.01)
        return B(a) * np.exp(-np.sum(B(steps)) * 0.01)
    
    M = max([f_density(val) for val in x_test]) * 1.1

    while len(samples) < n_samples:
        a_cand = np.random.uniform(0, a_max) 
        y_cand = np.random.uniform(0, M)
        
        if y_cand <= f_density(a_cand):
            samples.append(a_cand)
            
    return np.array(samples)

# Synthetic data generated with 
def B_power(a):
    return a**1.1 

N=2000
synthetic_ages = simulate_lineage(B_power, N, a_max=45)
np.savetxt("data/lin_synthetic_ages.txt", synthetic_ages)


PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"
lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])
real_ages = lin["ad"]

#Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#real data
ax1.hist(real_ages, bins=40, density=True)
ax1.set_title("Real ages distribution")
ax1.set_xlabel("Age at division")
ax1.set_ylabel("Density")

# synthetic data
ax2.hist(synthetic_ages, bins=40, density=True)
ax2.set_title("Synthetic ages distribution for $B(a)=a**2$")
ax2.set_xlabel("Age at division")

plt.tight_layout()
plt.savefig('outputs/synthetic_ages.png')