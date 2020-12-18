import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)

m0 = 5
m = 5
N = 1000
G = nx.barabasi_albert_graph(N, m)


# noinspection PyShadowingNames
def k_nn(g, k_max):
    knn = np.zeros(k_max)
    degrees = list(dict(G.degree()).values())
    degrees = np.array(degrees)
    knn_i = np.array(list(nx.average_neighbor_degree(G).values()))
    for k in range(k_max):
        delta_ki_k = np.equal(degrees, k).astype(int)
        numerator = np.sum(delta_ki_k*knn_i)
        denominator = np.sum(delta_ki_k)
        if denominator != 0:
            knn[k] = numerator/denominator
    return knn


realisations = 20
k_max = 200
M = np.zeros((k_max, realisations))
for i in range(realisations):
    G = nx.barabasi_albert_graph(N, m)
    knn_final = k_nn(G, k_max)
    for k in range(len(knn_final)):
        if knn_final[k] == 0:
            knn_final[k] = None
    M[:, i] = knn_final

Average = np.mean(M, axis=1)

plt.figure(figsize=(5, 5))
plt.plot(range(k_max), Average, 'x')
plt.xlabel('k, Degree of node', fontsize=16)
plt.ylabel(r'$k_{nn}(k)$ ', fontsize=16)
plt.savefig("knn.png")
