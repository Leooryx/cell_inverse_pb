import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PATH_POP = "pop_Eric0822_MDJ_sb_sd_ad.txt"
PATH_LIN = "lin_Lydia2901_new_MDJ_ad_sb_sd.txt"

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