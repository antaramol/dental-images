#%%
import matplotlib.pyplot as plt
import numpy as np


#%%
fig, ax = plt.subplots(1,3, figsize=(15,15))

channel = ['red', 'green', 'blue']

for i, val in enumerate(channel):
    arr = np.ones((100,100,3), dtype=np.uint8)
    arr[:,:,i] = arr[:,:,i]*20
    ax[i].imshow(arr)
    ax[i].set_title(val)
    ax[i].axis('off')

# %%
