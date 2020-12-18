import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statistics
np.random.seed(1234)

num_realizations = 20
N = 1000
zs = np.arange(0.1, 3.1, 0.1)
Cs = np.empty(zs.shape)

for i in range(len(zs)):
    z = zs[i]
    p = z / N
    Cs_N = np.empty((num_realizations,))

    for j in range(num_realizations):
        G = nx.gnp_random_graph(N, p, directed=True)
        Cs_N[j] = nx.average_clustering(G)

    Cs[i] = statistics.mean(Cs_N)

plt.plot(zs, Cs)
plt.xlabel("z")
plt.ylabel("C")
plt.tight_layout()
plt.savefig("Q3-3-B.png")
