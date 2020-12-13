import numpy as np
import matplotlib.pyplot as plt
import sdeint

np.random.seed(1234)

alpha = 1
sigma = 1
x_0 = 5
t_max = 10

dts = [0.1, 0.01]
colors = ["skyblue", "violet"]


def f(x, t):
    return -alpha*x


def g(x, t):
    return sigma*np.sin(t)


plt.figure(figsize=(20, 8))

for i in range(0, len(dts)):
    dt = dts[i]
    color = colors[i]

    times = np.arange(0, t_max, dt)
    result = sdeint.itoint(f, g, x_0, times)

    label = r"$\Delta t = " + str(dt) + "$"
    plt.plot(times, result, color=color, label=label)


plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$X_t$')
plt.title(r'Ornstein-Uhlenbeck process with $\alpha$ = {}, $\sigma$ = {}.'.format(alpha, sigma))
plt.savefig("Ornstein_Uhlenbeck.png")
plt.show()
