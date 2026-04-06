import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PATH_POP = "data/pop_Eric0822_MDJ_sb_sd_ad.txt"
PATH_LIN = "data/lin_Lydia2901_new_MDJ_ad_sb_sd.txt"

pop = pd.read_csv(PATH_POP, header=None, names=["sb", "sd", "ad"])
lin = pd.read_csv(PATH_LIN, header=None, names=["ad", "sb", "sd"])

# we need to reorder colums for coherence
pop = pop[["sb", "sd", "ad"]]
lin = lin[["sb", "sd", "ad"]]

pop["dataset"] = "Population"
lin["dataset"] = "Lineage"
df = pd.concat([pop, lin], ignore_index=True)

print("Statistical description")
print("For population:")
print(pop.describe().round(3))
print("\n")
print("For lineage:")
print(lin.describe().round(3))


# ________________________________
# 1. Variables distributions
# ________________________________
 
fig, axes = plt.subplots(2, 3, figsize=(13,7))
fig.suptitle("Variables distributions", fontweight="bold")
colors = {"Population": 'blue', "Lineage": 'green'}
var_labels = {"sb": "size at birth", "sd": "Size at division", "ad": "Age at division"}

for col_idx, (var, label) in enumerate(var_labels.items()):
    for row_idx, (name, data) in enumerate([("Population", pop), ("Lineage", lin)]):
        ax = axes[row_idx, col_idx]
        vals = data[var]
        ax.hist(vals, bins=40, color=colors[name], alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(vals.mean(), color="black", linestyle="--", label=f"{vals.mean():.2f}")
        ax.set_title(f"{name} - {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.tick_params()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_distributions.png")
plt.close()


# ______________________
# Relations between variables
# ______________________

pairs = [("sb", "sd"), ("sb", "ad"), ("sd", "ad")]
pair_labels = {
    ("sb","sd"): ("Size at birth", "Size at division"),
    ("sb","ad"): ("Size at birth", "Age at division"),
    ("sd","ad"): ("Size at division", "Age at division"),
}

fig, axes = plt.subplots(2,3, figsize=(13,7))
fig.suptitle("Relations between variables")

for col_idx, (vx, vy) in enumerate(pairs):
    for row_idx, (name, data) in enumerate([("Population", pop), ("Lineage", lin)]):
        ax = axes[row_idx, col_idx]
        x, y = data[vx], data[vy]
        common = data[[vx, vy]]
        ax.scatter(common[vx], common[vy], alpha=0.5, s=5, color=colors[name])
        ax.set_xlabel(pair_labels[(vx,vy)][0])
        ax.set_ylabel(pair_labels[(vx, vy)][1])
        ax.set_title(f"{name}")
        ax.legend()
        ax.tick_params()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_relations.png")
plt.close()


# _____________________
# Estimation of the size growth rate

# sd = sb*exp(\lambda*ad) --> len(sd/sb) = \lambda * ad 
# _____________________

lambdas = {}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Estimation of $\lambda$ (size growth rate) by regression\n" "$\ln(s_d/s_d) = \lambda a_d$") #can separate two terminated str sequences with a simple space

for ax, (name, data) in zip(axes, [("Lineage", lin), ("Population", pop)]):
    
    Y = np.log(data["sd"]/data["sb"])
    X = data["ad"].values
    
    # computations without intercept
    lam = np.dot(X, Y)/np.dot(X, X)
    Y_pred = lam * X
    ss_res = np.sum((Y - Y_pred)**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    # computations with intercept, using the stats package
    lambda_intercept, intercept, r_intercept, *_ = stats.linregress(X,Y)

    lambdas[name] = lam
    print(name)
    print(f"$\lambda$ (sans intercept) = {lam:.3f}, R^2 = {r2:.3f}")
    print(f"$\lambda$ (avec intercept) = {lambda_intercept:.3f}, intercept = {intercept:.3f}, R^2 = {r_intercept:.3f}")
    
    ax.scatter(X, Y, alpha=0.3, s=5, color=colors[name], label="data")
    xfit = np.linspace(0, X.max(), 200)
    ax.plot(xfit, lam * xfit, label=f"$\lambda$ (sans intercept) = {lam:.3f} (R^2 = {r2:.3f})")
    ax.plot(xfit, lambda_intercept * xfit, color="gray", linestyle='--', label=f"$\lambda$ (avec intercept) = {lambda_intercept:.3f} (R^2 = {r_intercept:.3f})")
    ax.set_xlabel("Age at division ($a_d$)")
    ax.set_ylabel("$\ln(s_d / s_b)$")
    ax.set_title(name)
    ax.legend()
    ax.tick_params()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_regression.png")
plt.close()




