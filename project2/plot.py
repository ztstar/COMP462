import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import trimesh
import alg
import utils

mesh_path = "./meshes/%s.stl" % "bunny"
mesh = trimesh.load(mesh_path)
 
# Sample 1000 stable grasps
num_samples = 1000
qualities = []
grasps = []

good = 0

for _ in range(num_samples):
    grasp, Q = alg.sample_stable_grasp(mesh, thresh=-1)
    grasps.append(grasp)
    qualities.append(Q)

    if Q > 0.018915:
        good += 1

qualities = np.array(qualities)

# Print summary statistics
print(f"Number of samples: {num_samples}")
print(f"Mean quality: {qualities.mean():.6f}")
print(f"Std quality:  {qualities.std():.6f}")
print(f"Min quality:  {qualities.min():.6f}")
print(f"Max quality:  {qualities.max():.6f}")
print(f"Good: {good}")

# Print first five sampled grasps
print("\nFirst five sampled grasps:")
for i in range(min(5, num_samples)):
    print(f"Grasp {i+1}: {grasps[i]}, Q = {qualities[i]:.6f}")

# Draw histogram
plt.figure(figsize=(8, 5))
plt.hist(qualities, bins=30, edgecolor='black')
plt.xlabel("Grasp Quality Q")
plt.ylabel("Frequency")
plt.title("Histogram of Stable Grasp Qualities")
plt.tight_layout()
plt.show()