import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches

a = np.sort(np.random.rand(6,18), axis=0)
x = np.arange(len(a[0]))

def alpha(i, base=0.2):
    l = lambda x: x+base-x*base
    ar = [l(0)]
    for j in range(i):
        ar.append(l(ar[-1]))
    return ar[-1]

fig, ax = plt.subplots(figsize=(4,2))

handles = []
labels=[]
for i in range(int(len(a)/2)):
    ax.fill_between(x, a[i, :], a[len(a)-1-i, :], color="blue", alpha=0.2)
    # handle = matplotlib.patches.Rectangle((0,0),1,1,color="blue", alpha=alpha(i, base=0.2))
    # handles.append(handle)
    # label = "quant {:.1f} to {:.1f}".format(float(i)/len(a)*100, 100-float(i)/len(a)*100)
    # labels.append(label)

handle1 = matplotlib.patches.Rectangle((0,0),1,1,color='#0343df', alpha=0.4)
handle2 = matplotlib.patches.Rectangle((0,0),1,1,color='#ff000d', alpha=0.4)
handles.append(handle1)
handles.append(handle2)
handles.append((handle1, handle2))
labels='pepe'

plt.legend(handles=handles, labels=labels, framealpha=1)
plt.show()