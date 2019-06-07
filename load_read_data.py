import scipy.io as sio
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=sio.loadmat('/Users/johannes/Documents/GitHub/Linearly-Constrained-NN/real_data/magnetic_field_data.mat')

pos = data['pos']
mag = data['mag']

n = np.size(pos, 0)  # number of data points

# normalising inputs (would this effect the constraints)???
min_x = pos[:,0].min()
min_y = pos[:,1].min()
min_z = pos[:,2].min()

max_x = pos[:,0].max()
max_y = pos[:,1].max()
max_z = pos[:,2].max()

X = pos.copy()
X[:,0] = (X[:,0]-min_x)/(max_x-min_x)*2.0 - 1.0
X[:,1] = (X[:,1]-min_y)/(max_y-min_y)*2.0 - 1.0
X[:,2] = (X[:,2]-min_z)/(max_z-min_z)*2.0 - 1.0

# how would you normalise the outputs?
# learnable scale?


# grid_x, grid_y= np.meshgrid(np.arange(-1.0, 1.0, 0.2),
#                       np.arange(-1.0, 1.0, 0.2))
# grid_z = -0.5*np.ones(np.shape(grid_x))

grid_x, grid_z= np.meshgrid(np.arange(-1.0, 1.0, 0.2),
                      np.arange(-1.0, 1.0, 0.2))
grid_y = 0.0*np.ones(np.shape(grid_x))

# grid_x, grid_y, grid_z= np.meshgrid(np.arange(-1.0, 1.0, 0.2),
#                       np.arange(-1.0, 1.0, 0.2),
#                     np.arange(-1.0, 1.0, 0.5))


mag_x_interp = griddata(X, mag[:,0], (grid_x, grid_y, grid_z), method='linear')
mag_y_interp = griddata(X, mag[:,1], (grid_x, grid_y, grid_z), method='linear')
# why does this one produce a nan if using linear?
mag_z_interp = griddata(X, mag[:,2], (grid_x, grid_y, grid_z), method='nearest')

# # with torch.no_grad():
# # Initialize plot
# # f, ax = plt.subplots(1, 1, figsize=(4, 4))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # ax.quiver(grid_x, grid_y, mag_x_interp, mag_y_interp)
# ax.quiver(grid_x, grid_y, grid_z, mag_x_interp, mag_y_interp, mag_z_interp,length=0.05, normalize=True)
# plt.show()

# with torch.no_grad():
# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.quiver(grid_x, grid_z, mag_x_interp, mag_y_interp)
plt.show()