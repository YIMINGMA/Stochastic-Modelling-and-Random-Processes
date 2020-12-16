import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma = 1

mu1 = -0.5
mu2 = -1
mu3 = 1
mus = [mu1, mu2, mu3]

dt1 = 0.1
dt2 = 0.01
dts = [dt1, dt2]


def drift(mu, y):
    return (mu + 0.5*sigma**2) * y


def noise(y):
    return sigma*y


plt.figure(figsize=(15, 15))
for j in range(len(mus)):
    plt.subplot(len(mus), 1, j+1)
    for dt in dts:
        ts = np.arange(0, 10+dt, step=dt)
        Ys = np.empty(ts.shape)
        Ys[0] = 1
        for i in range(len(Ys) - 1):
            Ys[i+1] = Ys[i] + drift(mus[j], Ys[i])*dt + noise(Ys[i])*np.random.normal(0, dt)
        plt.plot(ts, Ys, label=r"$\Delta t = %.2f$" % dt)
    plt.legend()
    plt.title(r"$Y_t$ when $\mu = $%.1f" % mus[j])
    plt.xlim(0, 10)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$Y_t$")

plt.tight_layout()
plt.savefig("Q1-D.png")
