import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg as la
rng = np.random.RandomState(1234)

t = np.linspace(0,20,1000)
data_spring = np.zeros((3,len(t)))
data_spring[0] = np.sin(2*t) # frequency of 2, amplitude of 1; setting the x values 
data_spring[1] = rng.normal(loc=0,scale=0.1,size=len(t)) # y values drawn from a gaussian

plt.plot(t,data_spring[0])
plt.show()


def get_rotation_matrix(theta,phi):
    from scipy.spatial.transform import Rotation # we will use the Rotation class from scipy in order to change basis
    r1 = Rotation.from_euler('Z',-theta,degrees=True)
    r2 = Rotation.from_euler('Y',-phi,degrees=True)
    r = r2.as_matrix() @ r1.as_matrix()
    return r


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(-1.5,1.5)

ax.plot(data_spring[0,:],data_spring[1,:],data_spring[2,:])


n = 10 # number of cameras
colors = plt.cm.twilight_shifted(np.linspace(0, 1, n))
theta = np.linspace(0,360,n)
phi = np.linspace(0,90,n)

for i in range(n):
    r = get_rotation_matrix(theta[i],phi[i])
    data_rotated = r @ data_spring
    ax.plot(data_rotated[0,:],data_rotated[1,:],data_rotated[2,:],color=colors[i],alpha=0.4)


plt.show()






