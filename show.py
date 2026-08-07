'''plot'''

import matplotlib.pyplot as plt

plt.plot(np.arange(len(TE_omega)), TE_omega, 'b.', markersize=1, label='TE')
plt.plot(np.arange(len(TM_omega)), TM_omega, 'r.', markersize=1, label='TM')
plt.xlabel('k-point')
plt.ylabel('Frequency (rad/ps)')
plt.legend()
plt.show()
