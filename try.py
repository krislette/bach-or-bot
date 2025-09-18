import numpy as np

data = np.load("data/processed/training_data.npz")
data2 = np.load("data/processed/training_data2.npz")

print(data["X"])
print(data2["X"])