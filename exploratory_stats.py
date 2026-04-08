import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
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
# 2. Relations between variables
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
# 3. Estimation of the size growth rate

# sd = sb*exp(\lambda*ad) --> len(sd/sb) = \lambda * ad 
# _____________________

results = {
    "Lineage": {},
    "Population": {}
}

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("Estimation of g (size growth rate) by regression: $ln(s_d/s_d) = g*a_d$") 

for ax, (name, data) in zip(axes, [("Lineage", lin), ("Population", pop)]):
    
    Y = np.log(data["sd"]/data["sb"])
    X = data["ad"].values
    
    # computations without intercept
    model_1 = sm.OLS(Y,X).fit()
    g_1 = model_1.params[0]
    p_value_1 = model_1.pvalues[0]
    r2_1 = model_1.rsquared

    # computations with intercept, using the stats package
    X_intercept = sm.add_constant(X)   # adds intercept
    model_2 = sm.OLS(Y, X_intercept).fit()

    g_2 = model_2.params[1]
    intercept = model_2.params[0]
    p_value_2 = model_2.pvalues[1]
    r2_2 = model_2.rsquared

    results[name] = {
        "g (no intercept)": g_1,
        "R² (no intercept)": r2_1,
        "p-value (no intercept)": p_value_1,
        "g (with intercept)": g_2,
        "intercept": intercept,
        "R² (with intercept)": r2_2,
        "p-value (with intercept)": p_value_2,
    }
    
    print(f"g (without intercept) = {g_1:.3f}, R^2 = {r2_1:.3f}, p_value = {p_value_1:.5f}")
    print(f"g (with intercept) = {g_2:.3f}, intercept = {intercept:.3f}, R^2 = {r2_2:.3f}, p_value = {p_value_2:.5f}")
    
    ax.scatter(X, Y, alpha=0.3, s=5, color=colors[name], label="data")
    xfit = np.linspace(0, X.max(), 200)
    ax.plot(xfit, g_1 * xfit, label=f"Without intercept: g={g_1:.3f}")
    ax.plot(xfit, g_2 * xfit, color="gray", linestyle='--', label=f"With intercept: g={g_2:.3f}")
    ax.set_xlabel(r"Age at division ($a_d$)")
    ax.set_ylabel(r"$ln(s_d / s_b)$")
    ax.set_title(f"{name}")
    ax.legend()
    ax.tick_params()



plt.tight_layout() 
plt.savefig(f"{OUTPUT_DIR}/3_regression.png")
plt.close()

# table for clean illustration
fig, ax = plt.subplots(figsize=(8, 5))  
ax.axis('off')  # remove axes

row_labels = [
    "g (no intercept)",
    "R² (no intercept)",
    "p-value (no intercept)",
    "g (with intercept)",
    "intercept",
    "R² (with intercept)",
    "p-value (with intercept)",
]

col_labels = ["Lineage", "Population"]

table_data = []
for row in row_labels:
    table_data.append([
        f"{results['Lineage'][row]:.3e}" if "p-value" in row else f"{results['Lineage'][row]:.3f}",
        f"{results['Population'][row]:.3e}" if "p-value" in row else f"{results['Population'][row]:.3f}",
    ])

table = ax.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='center',     
    cellLoc='center',
    rowLoc='center'
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_regtable.png")
plt.close()