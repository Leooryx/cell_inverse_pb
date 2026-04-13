import numpy as np
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



def sample_division_size(X_current, B, B_max):
    x = X_current 
    while True:
        x += np.random.exponential(1/B_max)
        if np.random.rand() < B(x)/B_max:
            return x


def simulate_lineage_size(B, grid, num_samples, growth_rate, Xbar):
    B_arr = _to_array(B,grid)
    B_max=np.max(B_arr)*1.1
    
    A = []
    Xb = []
    Xd = []
    X_current = Xbar #initialisation

    for _ in range(num_samples):
        X_div = sample_division_size(X_current, B, B_max)
        A_div = (1/growth_rate)*np.log(X_div / X_current)
        A.append(np.round(A_div, 3))
        Xb.append(np.round(X_current, 3))
        Xd.append(np.round(X_div, 3))
        X_current = X_div / 2
    
    return np.column_stack((A, Xb, Xd))




if __name__ == '__main__':

    import matplotlib.pyplot as plt

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
    growth_rate = 0.545

    def B_power(a):
            return a**power

    if test_age:
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
        plt.close()


    if test_size:
        s_max = np.max(real_Xd)
        grid = np.linspace(0, s_max, 2000)
        synthetic_size_data = simulate_lineage_size(B_power, grid, N, growth_rate, Xbar)
        np.savetxt("data/synthetic_lin_size_model.txt", synthetic_size_data, delimiter=",")
        synthetic_A = synthetic_size_data[:,0]
        synthetic_Xb = synthetic_size_data[:,1]
        synthetic_Xd = synthetic_size_data[:,2]
        synthetic_A_max = np.round(np.max(synthetic_A),3)
        synthetic_Xbar = np.round(np.mean(synthetic_Xb), 3)
        print("real a_max:", a_max, "/ real X_bar:", Xbar)
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
        axes[0,1].set_title(f"Synthetic ages distribution for B(x)=x**{power}")
        axes[0,1].set_xlabel("Age at division")
        axes[0,1].grid()

        axes[1,1].hist(synthetic_Xb, bins=40, density=True)
        axes[1,1].set_title(f"Synthetic sizes at birth distribution for B(x)=x**{power}")
        axes[1,1].set_xlabel("Size at birth")
        axes[1,1].grid()

        axes[2,1].hist(synthetic_Xd, bins=40, density=True)
        axes[2,1].set_title(f"Synthetic sizes at division distribution for B(x)=x**{power}")
        axes[2,1].set_xlabel("Size at division")
        axes[2,1].grid()

        plt.tight_layout()
        plt.savefig('outputs/synthetic_lin_size_model.png')
        plt.close()

