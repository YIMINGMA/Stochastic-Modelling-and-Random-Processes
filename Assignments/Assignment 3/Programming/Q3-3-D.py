import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
np.random.seed(1234)

zs = [0.5, 1.5, 5, 10]
num_realizations = 20
N = 1000


# noinspection PyShadowingNames
def semicircle(sigma, mu, λ):
    wigner = np.zeros(len(λ))
    for j in range(len(λ)):
        if abs(λ[j]) < 2*sigma:
            wigner[j] = (1 / (2 * np.pi * sigma**2)) * np.sqrt(4 * sigma**2 - λ[j]**2) + mu
        else:
            wigner[j] = 0
    return wigner


# noinspection PyShadowingNames
def plot(p, num_realizations):
    As = []
    evals = []
    for i in range(num_realizations):
        G = nx.gnp_random_graph(N, p, directed=True)
        A = nx.to_numpy_matrix(G)
        As.append(A)
        evals_temp, evecs_temp = np.linalg.eig(A / np.sqrt(N))
        evals.append(evals_temp)

    # noinspection DuplicatedCode
    evals = np.array(evals)
    evals = evals.flatten()
    evals = evals.real
    spectral_density = stats.gaussian_kde(evals, bw_method=0.05)
    λs = np.linspace(min(evals), max(evals), N * num_realizations)
    plt.plot(λs, abs(spectral_density(λs)), label="Kernel Density Estimation")
    plt.xlabel('λ')
    plt.ylabel('ρ(λ)')
    As = np.array(As)
    As = As.flatten()
    σ = np.std(As)
    μ = np.mean(As)
    wigner = semicircle(σ, μ, λs)
    plt.plot(λs, wigner, label="Wigner Semi-Circle Law")
    plt.legend()


plt.figure(figsize=(15, 16))
plt.subplot(411)
z = zs[0]
p = z / N
plot(p, num_realizations)

plt.subplot(412)
z = zs[1]
p = z / N
plot(p, num_realizations)

plt.subplot(413)
z = zs[2]
p = z / N
plot(p, num_realizations)

plt.subplot(414)
z = zs[3]
p = z / N
plot(p, num_realizations)

plt.tight_layout()
plt.savefig("Q3-3-D.png")
