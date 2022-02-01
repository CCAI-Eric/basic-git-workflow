import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def linearFunc(net, m=1):
    return m * net

def tanh(net):
    return np.tanh(net)

def logisticfunc(net, c = 1):
    return 1 / (1+ np.exp(-c*net))

def stepfunc(net):
    return np.heaviside(net, np.nan)

def relu_func(net):
    # return np.piecewise(net, [net < -t, net > t], [0, 1, lambda net: 0.5 + net / (2 * t)])
    return np.maximum(0, net)

def Sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

list = np.linspace(-3, 3, endpoint=True)
print(list)

# seaborn sets
sns.set_theme()
# sns.set_style('darkgrid', {"axes.facecolor": ".1"})
# sns.set_style('ticks', {"xtick.major.size": ".1"})
# sns.axes_style('darkgrid')
# sns.set_style('darkgrid')
n = 3
f, ax = plt.subplots(figsize=(10, 6))
plt.plot(list, tanh(list), label='Tanh', linewidth=n)
plt.plot(list, Sigmoid(list), label='Sigmoid', linewidth=n)
plt.plot(list, linearFunc(list, 1), '--', label='Linear', linewidth=n)
plt.plot(list, stepfunc(list), label='Step / Heavyside', linewidth=n)
plt.plot(list, relu_func(list), '-.', label='ReLU', linewidth=n)
plt.grid()
plt.title("Aktivierungsfunktionen", size=20)
plt.xticks([-1, 1])
plt.yticks([1, -1])
plt.xlim(-3, 3)
plt.ylim(-1.2, 1.2)
plt.legend(fontsize=14)
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# ax.spines['left'].set_color('black')

ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
#ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_ticks_position('left')
plt.savefig('act_2.png', dpi=400)
plt.show()