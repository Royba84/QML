# Python Code in order to execute the SVM demonstration graphs:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC

# Plot 1: Linear Separation
plt.figure(figsize=(12, 4))

# Linearly Separable Data
X, y = make_blobs(n_samples=50, centers=2, random_state=6)
svm_linear = SVC(kernel='linear', C=1E6).fit(X, y)
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Hyperplane
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
Z = svm_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("1. Linear Separation")

# Plot 2: Support Vectors and Margin
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(svm_linear.support_vectors_[:, 0], svm_linear.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("2. Support Vectors and Margin")

# Plot 3: Kernel Trick
X, y = make_circles(100, factor=.1, noise=.1)
svm_rbf = SVC(kernel='rbf', C=1E6).fit(X, y)
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Hyperplane
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("3. Kernel Trick")

plt.tight_layout()
plt.show()

