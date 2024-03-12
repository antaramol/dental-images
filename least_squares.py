#%%

# plot some random points in a 2d plane
import numpy as np
import matplotlib.pyplot as plt

# generate random points 
n = 10
x = np.random.rand(n)
y = np.random.rand(n)

# plot the points
plt.scatter(x, y)

plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %%

# use least squares to fit a line to the points
# start with M = 1 (a line) to M = 9

# create the matrix X
# plot the points
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")


# plot every solution for M, one at a time

for M in [2,10]:
    X = np.zeros((n, M + 1))
    for i in range(M + 1):
        X[:, i] = x ** i
    w = np.linalg.solve(X.T @ X, X.T @ y)
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.zeros(100)
    for i in range(M + 1):
        y_vals += w[i] * x_vals ** i
    plt.plot(x_vals, y_vals, label=f"M={M}")


# plot y axis between 0 and 4
plt.ylim(0,1)
plt.legend()
plt.show()


# %%
