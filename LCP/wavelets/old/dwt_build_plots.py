# This script generates some plots used in the notebook.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
# plt.style.use('seaborn')

# --- first plot
x=np.arange(1,17)
y=np.sin(0.5*x)+np.random.normal(size=len(x))

x_m = [(x[i]+x[i+1])/2 for i in range(0,len(x)-1,2)]
y_m = [(y[i]+y[i+1])/2 for i in range(0,len(y)-1,2)]

fig,ax=plt.subplots()
ax.step(x,y,linestyle='--',where='post',color='pink',alpha=0.7)
ax.step(x_m,y_m,linestyle='--',where='post',color='palegreen')
for i in range(0,len(x),2):
    ax.plot([x[i],x[i+1]],[y[i],y[i+1]],linestyle='-',color='lightgreen')
    
ax.plot(x_m,y_m,'go',label='Averaged signal')
ax.plot(x,y,'ro',label='Original signal data points')
ax.xaxis.set_major_locator(ticker.FixedLocator(x))
ax.legend()
plt.savefig("./dwt_plots/original_signal_with_average.png")
plt.show()

# haar wavelet (psi) and scaling (phi) function
x_h_psi = [0.,0.,0.5,0.5,1.,1.]
y_h_psi = [0.,1.,1.,-1.,-1.,0.]
x_h_phi = [0.,0.,1.,1.]
y_h_phi = [0.,1.,1.,0.]
fig,axs = plt.subplots(1,2,figsize=[10,4])
axs[0].plot(x_h_phi,y_h_phi)
axs[0].set_title('Haar Scaling Function $\phi$')
axs[1].plot(x_h_psi,y_h_psi,color='navy')
axs[1].set_title('Haar Wavelet Function $\psi$')
plt.savefig("./dwt_plots/haar_phi_psi.png")
plt.show()