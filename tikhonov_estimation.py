import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_LIN = "data/distribution-age-taille_oldpole_0210.dat"
lin = pd.read_csv(PATH_LIN, header=0, names=["index", "age", "size"], sep=" ")
sizes = lin["size"]

"""fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(sizes, bins=50)
ax.grid(True)
ax.set_title("Real sizes distribution")
ax.set_xlabel("Sizes at all times")
ax.set_ylabel("Frequency") 
plt.savefig("outputs/sizes_all_times.png")"""


indices = lin[["index"]].astype(int)
values = lin[["size"]]

print(values.describe())
