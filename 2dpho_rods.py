'''DOS 없는 구현 (rods 버전: 사각 격자 위에 원형 로드)'''

import os
import numpy as np
from scipy.special import jv
from datetime import datetime
from parameters import a, r, eps1, eps2, Gmf, Nx, c

# Shared parameters are defined in parameters.py.
gamma1=eps1**-1
gamma2=eps2**-1

dk=np.pi/(Nx*a)
dG=2*np.pi/a # 고유값은 nm**-2 꼴

'''(0~np.pi/a,0) 감마X, (0,0~np.pi/a) XM, (0~np.pi/a,0~np.pi/a) X감마 -> k그리드 생성. 각 k그리드에 대해서 수행하는거'''

# Γ → X:
# (0,0), (dk,0), ..., (Nx·dk,0)
n = np.arange(Nx + 1)
K_GX = np.column_stack((
    n * dk,
    np.zeros(Nx + 1)
))

# X → M:
# (Nx·dk,dk), ..., (Nx·dk,Nx·dk)
# X 중복 방지
n = np.arange(1, Nx + 1)
K_XM = np.column_stack((
    np.full(Nx, Nx * dk),
    n * dk
))

# M → Γ:
# ((Nx-1)dk,(Nx-1)dk), ..., (0,0)
# M 중복 방지
n = np.arange(Nx - 1, -1, -1)
K_MG = np.column_stack((
    n * dk,
    n * dk
))

# shape: (3*Nx + 1, 2)
# 각 행이 하나의 k = [kx, ky]. 2d array라서 열벡터를 stack한 꼴임.
K = np.vstack((K_GX, K_XM, K_MG))


'''감마, G 행렬 성분 정의 (벡터화 버전, 원형 로드 형상factor)

veins 버전은 1차원 슬릿 폭(b/a)의 sinc 형태였지만, rods 버전은 반지름 r인 원형
산란체의 2차원 푸리에 변환이라 1차 베셀함수 J1이 등장한다.

gamma_{m,n} = gamma1 + (gamma2-gamma1)*pi*(r/a)^2                                   , m=0 and n=0
            = (gamma2-gamma1) * (r/a) / rho * J1(2*pi*(r/a)*rho), rho=sqrt(m^2+n^2)  , otherwise
'''

def gamma_matrix(delta_m, delta_n):
    is_00 = (delta_m == 0) & (delta_n == 0)
    rho = np.sqrt(delta_m**2 + delta_n**2)
    safe_rho = np.where(is_00, 1, rho)  # 0으로 나누기 방지용(해당 위치 결과는 안 씀)

    val00 = gamma1 + (gamma2 - gamma1) * np.pi * (r / a)**2
    valMN = (gamma2 - gamma1) * (r / a) / safe_rho * jv(1, 2 * np.pi * (r / a) * safe_rho)

    return np.where(is_00, val00, valMN)

'''max를 충족하는 m, n 리스트업'''
Gpoint_list = []

for m in range(-int(Gmf), int(Gmf) + 1):
    for n in range(-int(Gmf), int(Gmf) + 1):
        if m**2 + n**2 < Gmf**2:
            Gpoint_list.append((m, n))

Gpts = np.array(Gpoint_list)   # (N, 2) 정수 (m, n) 목록
Gvecs = Gpts * dG              # (N, 2) 각 (m, n)에 대응하는 G벡터

# gamma_mn(o-m, p-n)을 모든 (i, j) 쌍에 대해 한 번에 계산 (k와 무관하므로 루프 밖에서 1회만)
delta_m = Gpts[:, 0][:, None] - Gpts[:, 0][None, :]
delta_n = Gpts[:, 1][:, None] - Gpts[:, 1][None, :]
Gamma = gamma_matrix(delta_m, delta_n)  # (N, N)

'''k 받고 A, B 행렬 반환하는 함수 정의'''
def A_B_at_k(kvec):
    q = kvec + Gvecs                      # (N, 2) 이 k에서의 q벡터들
    norms = np.linalg.norm(q, axis=1)     # (N,)
    A = Gamma * np.outer(norms, norms)    # A[i,j] = Gamma[i,j] * |q_i| * |q_j|
    B = Gamma * (q @ q.T)                 # B[i,j] = Gamma[i,j] * (q_i · q_j)
    return A, B

'''고유값 획득 루틴'''
TE_omega = []
TM_omega = []

for kvec in K:
    A, B = A_B_at_k(kvec)
    eig_TE = np.linalg.eigvalsh(A)
    eig_TM = np.linalg.eigvalsh(B)
    omega_TE=c*np.sqrt(abs(eig_TE))
    omega_TM=c*np.sqrt(abs(eig_TM))
    TE_omega.append(omega_TE)
    TM_omega.append(omega_TM)

#지금 rad*THz 단위인데, 격자가 nm 스케일이라 xray로 나올 수 있어서 위 함수 리스케일링 하면 됨

# 각 행은 K의 한 k-vector에 대응한다.
# 열: k_index, kx, ky, TE band 0.., TM band 0..
TE_omega_array = np.array(TE_omega)
TM_omega_array = np.array(TM_omega)

'''결과 데이터랑 플롯 루틴을 분리하기 위한 저장 알고리즘(csv)'''
band_count = TE_omega_array.shape[1]
header = ["k_index", "kx", "ky"]
header += [f"TE_band_{j}" for j in range(band_count)]
header += [f"TM_band_{j}" for j in range(band_count)]

output_data = np.column_stack((
    np.arange(len(K)),
    K,
    TE_omega_array,
    TM_omega_array,
))

os.makedirs("python_csv", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = (
    f"python_csv/2dpho_rods_a={a:g}_r={r:g}_eps1={eps1:g}_eps2={eps2:g}_"
    f"Nx={Nx}_Gmf={Gmf:g}_{timestamp}.csv"
)

np.savetxt(
    output_filename,
    output_data,
    delimiter=",",
    header=",".join(header),
    comments="",
)

print(f"Saved eigenfrequency data to: {output_filename}")
