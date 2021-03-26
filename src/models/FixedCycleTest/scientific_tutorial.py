import numpy as np


#%%

x = np.linspace(0,10,1000)
y = np.sin(x)
print(y)


#%%
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.show()