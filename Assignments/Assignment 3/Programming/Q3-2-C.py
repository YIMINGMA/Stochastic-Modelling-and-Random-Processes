import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
np.random.seed(1234)

m0 = 5
m = 5
N = 1000

realisations = 20
As = []
evals = []
for i in range(realisations):
    G = nx.barabasi_albert_graph(N, m)
    A = nx.to_numpy_matrix(G)
    As.append(A)
    evals_temp, evecs_temp = np.linalg.eig(A/np.sqrt(N))
    evals.append(evals_temp)

evals = np.array(evals)
evals = evals.flatten()
spectral_density = stats.gaussian_kde(evals, bw_method=0.05)
λs = np.linspace(min(evals), max(evals), N*realisations)
plt.plot(λs, spectral_density(λs), label="Kernel Density Estimation")
plt.xlabel('λ')
plt.ylabel('ρ(λ)')


# noinspection PyShadowingNames
def semicircle(sigma, mu, λ):
    wigner = np.zeros(len(λ))
    for j in range(len(λ)):
        if abs(λ[j]) < 2*sigma:
            wigner[j] = (1 / (2 * np.pi * sigma**2)) * np.sqrt(4 * sigma**2 - λ[j]**2) + mu
        else:
            wigner[j] = 0
    return wigner


As = np.array(As)
As = As.flatten()
σ = np.std(As)
μ = np.mean(As)
wigner = semicircle(σ, μ, λs)
plt.plot(λs, wigner, label="Wigner Semi-Circle Law")
plt.legend()
plt.tight_layout()
plt.savefig("spectrum.png")
