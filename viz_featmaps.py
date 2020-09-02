import numpy as np 
import matplotlib.pyplot as plt 

path = "featmaps/featmaps_7500_5.npy"
featmaps = np.load(path)

rand_idxs = np.random.randint(0,featmaps.shape[1], 5)

for idx in rand_idxs:
    featmap = featmaps[0,idx,:,:]
    plt.imshow(featmap*0.5+0.5, cmap='gray')
    plt.show()
