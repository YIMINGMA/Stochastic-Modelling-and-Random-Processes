import numpy as np
import matplotlib.pyplot as plt

Ls = [10, 100, 1000]
colors = ["lightcoral", 'orange', 'cyan']
for i in range(0, len(Ls)):
    L = Ls[i]
    color = colors[i]
    time = 0
    WT = 0

    for n in range(L, 1, -1):
        waitTime = np.random.exponential(scale=2/(n*(n-1)))
        plt.plot([time/L, (time+waitTime)/L], [n/L, n/L], color=color, lw=2)
        time += waitTime
        WT = waitTime

    plt.plot([time/L, (time+2*WT)/L], [1/L, 1/L], color=color, label=r"$L = " + str(L) + "$")

plt.title("Kingman's Coalescent")
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$N_t$')
plt.yscale('linear')
plt.xscale('log')
plt.savefig("Kingman_Coalescent.png")
plt.show()
