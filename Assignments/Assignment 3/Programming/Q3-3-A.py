import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statistics
np.random.seed(1234)

num_realizations = 20
N1 = 100
N2 = 1000
Ns = [N1, N2]
zs = np.arange(0.1, 3.1, 0.1)


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


plt.figure(figsize=(15, 10))

for N in Ns:
    largest1 = []
    largest2 = []

    for z in zs:
        p = z / N
        largest1_realization = np.empty((num_realizations,))
        largest2_realization = np.empty((num_realizations,))
        
        for i in range(num_realizations):
            G = nx.gnp_random_graph(N, p, directed=True)
            ks, Pk = degree_distribution(G)
            Pk = list(Pk)
            Pk.sort(reverse=True)
            largest1_realization[i] = Pk[0]
            largest2_realization[i] = Pk[1]
        
        largest1.append(statistics.mean(largest1_realization))
        largest2.append(statistics.mean(largest2_realization))

    plt.plot(zs, largest1, label="N=%d, the largest component" % N)
    plt.plot(zs, largest2, label="N=%d, the 2nd largest component" % N)

plt.legend()
plt.tight_layout()
plt.savefig("Q3-3-A.png")
