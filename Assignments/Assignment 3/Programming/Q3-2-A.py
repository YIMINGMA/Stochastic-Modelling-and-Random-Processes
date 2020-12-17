import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
np.random.seed(1234)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

m0 = 5
m = 5
N = 1000
G = nx.barabasi_albert_graph(N, m)

pos = nx.fruchterman_reingold_layout(G)
plt.figure(figsize=(50, 50))
plt.axis("off")
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="black")
nx.draw_networkx_edges(G, pos, alpha=0.500)
nx.draw_networkx_labels(G, pos, font_color="white")
plt.tight_layout()
plt.savefig("simulation.png")


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


Count_degree = list(dict(G.degree()).values())
ks, Pk = degree_distribution(G)

plt.figure(figsize=(15, 5))
plt.plot(ks, Pk, 'bo', label='Data')
plt.xlabel(r"$k$", fontsize=20)
plt.ylabel(r"$P(k)$", fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("degree-distribution.png")

plt.figure(figsize=(15, 5))
cdf = np.cumsum(Pk)
x = np.linspace(np.min(Count_degree), np.max(Count_degree), 10000)
plt.plot(range(len(cdf)), 1-cdf, '--', label="One realisation, empirical tail")
plt.plot(x, x**(-2), label='power law')
plt.xlabel('degree, k', fontsize=16)
plt.title('Empirical Tail (1 realisation)', fontsize=16)
plt.xlim([np.min(Count_degree), np.max(Count_degree)])
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.legend(fontsize=12)
plt.savefig("1-twenty-realisations.png")

realisations = 20
degrees = []
for r in range(realisations):
    G = nx.barabasi_albert_graph(N, m)
    Count_degree = list(dict(G.degree()).values())
    degrees.append(Count_degree)
degrees = np.array(degrees)
degrees = degrees.flatten()
ecdf_deg = ECDF(degrees)

plt.figure(figsize=(15, 5))
plt.plot(ecdf_deg.x, np.ones(len(ecdf_deg.y))-ecdf_deg.y, '--', label="20 realisations, empirical tail")
Xx = np.linspace(min(degrees), max(degrees), 1000)
plt.plot(Xx, Xx**(-2), label='Power law')
plt.title('Empirical Tail (20 realisations)', fontsize=16)
plt.legend(fontsize=12)
plt.xlabel('degree, k', fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.xlim([np.min(degrees), np.max(degrees)])
plt.ylim([1e-6, 1])
plt.tight_layout()
plt.savefig("20-twenty-realisations.png")
