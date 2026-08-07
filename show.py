"""Plot TE and TM dispersion bands from the CSV produced by 2dpho.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import a, Nx, c

num_bands = 8
nu_max = 1.4

# Choose the exact result file to plot.
csv_filename = Path(
    r"C:\Users\cojyi\Desktop\BIC\2dpho_a=10_d=0_eps1=1_eps2=8.9_Nx=35_Gmf=15_20260807_124708.csv"
)
if not csv_filename.is_file():
    raise FileNotFoundError(f"CSV file not found: {csv_filename}")

data = np.loadtxt(csv_filename, delimiter=",", skiprows=1)

# print(data.shape) #(106, 1397=3+2*697)

kvecs = data[:, 1:3] #전체 행, 1 2 열
# print(kvecs.shape)  # (106, 2)
total_bands = (data.shape[1] - 3) // 2 #697
TE_omega = data[:, 3:3 + total_bands] #전체 행, TE 전체 밴드(열)
TM_omega = data[:, 3 + total_bands:3 + 2 * total_bands] # 전체 행, TM 전체 밴드(열)

TE_nu = TE_omega * a / (2 * np.pi * c)
TM_nu = TM_omega * a / (2 * np.pi * c) #\nu

# Cumulative distance along the Gamma-X-M-Gamma path.
step_lengths = np.linalg.norm(np.diff(kvecs, axis=0), axis=1) 
# 행 방향으로 빼고 열 방향으로 norm. 차이 벡터의 길이. dkx, dky, sqrt(dkx^2 + dky^2) 구해서. M-Gamma가 sqrt(2)배
k_path = np.concatenate(([0.0], np.cumsum(step_lengths)))
# [0.0]과 누적합 [distance1, distance1+distance2, ...] 합쳐서 k_path 축 생성 1+106-1꼴 완성

for band in range(num_bands):
    plt.plot(k_path, TM_nu[:, band], color="blue", label="TM" if band == 0 else None)
    plt.plot(k_path, TE_nu[:, band], color="red", label="TE" if band == 0 else None)
    #nu 값중 num_band 안의 밴드들을 플롯 후 저장해둠
    #0번째 밴드에만 labeling하기 위한 else None 처리.

symmetry_indices = [0, Nx, 2 * Nx, 3 * Nx]
symmetry_positions = k_path[symmetry_indices]
for position in symmetry_positions[1:-1]:
    plt.axvline(position, color="black", linewidth=0.8, alpha=0.5)
    # 첫 번째(0)와 마지막(감마) 제외한 곳에 axvline(x축에 수직한 선) 그리기.

plt.xticks(symmetry_positions, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
#symmetry_positions에 대한 xticks를 그리기. 라벨은 LaTeX로 그리기 위해 r"$\Gamma$" 등으로 처리.
plt.xlim(k_path[0], k_path[-1])
plt.ylim(0, nu_max)
plt.xlabel("Wave vector")
plt.ylabel(r"Normalized frequency $\nu = \omega a/(2\pi c)$")
plt.legend()
plt.tight_layout()
plt.show()
