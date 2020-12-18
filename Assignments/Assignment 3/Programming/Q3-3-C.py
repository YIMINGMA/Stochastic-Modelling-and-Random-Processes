import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
np.random.seed(1234)

num_realizations = 20
N = 1000
z = 1
p = z / N


# noinspection DuplicatedCode
def degree_distribution(ger):
    vk = dict(ger.degree())
    vk = list(vk.values())
    max_k = np.max(vk)
    k_values = np.arange(0, max_k+1)
    p_k = np.zeros(max_k+1)
    for k in vk:
        p_k[k] = p_k[k] + 1
    p_k = p_k / sum(p_k)
    return k_values, p_k


plt.figure(figsize=(15, 8))
Pks = []
for i in range(num_realizations):
    G = nx.gnp_random_graph(N, p, directed=True)
    ks, Pk = degree_distribution(G)
    plt.plot(ks, Pk, 'bo')
    Pks.append(Pk)

length = max(map(len, Pks))
mean_Pk = np.zeros((length,))
for i in range(len(mean_Pk)):
    counter = 0
    for j in range(len(Pks)):
        if len(Pks[j]) >= i+1:
            mean_Pk[i] = mean_Pk[i] + Pks[j][i]
            counter = counter+1
    mean_Pk[i] = mean_Pk[i] / counter
ks = range(0, length)
plt.plot(ks, mean_Pk, 'yo--', linewidth=2, markersize=12, label="Mean of Samples")
plt.plot(ks, poisson.pmf(ks, z), 'r+--', linewidth=2, markersize=12, label="Poisson Distribution")
plt.legend()
plt.xlabel("k")
plt.ylabel("p")
plt.tight_layout()
plt.savefig("Q3-3-C.png")
