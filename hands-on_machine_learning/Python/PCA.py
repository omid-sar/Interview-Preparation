import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 


pca = PCA(n_components=2)


# --------------------------- LLE (Localy Linear Embedding) ------------------------ #
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

X_swiss, t = make_swiss_roll(n_samples=10000, noise=0.2, random_state=42)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X_swiss[:,0], X_swiss[:,1], X_swiss[:,2], c=t, cmap='Spectral')
ax.set_title(" 3D Swiss Roll Dataset")
plt.show()


lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)

fig, ax  = plt.subplots(figsize=(10,6))
scatter= ax.scatter(X_unrolled[:,0], X_unrolled[:,1], c=t, cmap='Spectral')
ax.set_title(" UNROLLED 3D Swiss Roll Dataset")
plt.show()

# ----------------------------------- PCA Guassian --------------------------------- #
n_points = 10_000
xC = np.array([2, 1]) # 
sig = np.array([2, 0.5])
theta = np.pi/3
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

np.random.randn(2, n_points)
X = R @ np.diag(sig) @ np.random.randn(2, n_points) + np.diag(xC) @ np.ones((2,n_points))


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,2,1)
ax.scatter(X[0,:], X[1,:], color="k")
ax.grid()

X_avg = np.mean(X, axis=1)