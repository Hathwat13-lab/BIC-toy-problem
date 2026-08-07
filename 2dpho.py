'''DOS 없는 구현'''

import numpy as np
from datetime import datetime

'''변수 할당'''

a = 10 #nm
d = 0 #nm
b = a - d

Gmf = 15.0 # Gmax=dG*Gmf
Nx = 35

eps1=1.0
eps2=8.9
gamma1=eps1**-1
gamma2=eps2**-1

dk=np.pi/(Nx*a)
dG=2*np.pi/a # 고유값은 nm**-2 꼴 
c=3*10**5 #nm/ps...최종 c*sqrt(고유값) 하면 rad*THz 꼴

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

# print(K)
# print(K.shape)  # (3*Nx + 1, 2) +1 은 감마 포인트가 중복으로 세지는 점을 고려해서...


'''감마, G 행렬 성분 정의'''

def gamma_mn(m, n):
    if m == 0 and n == 0:
        return gamma2 + (gamma1 - gamma2) * (b / a)**2
    elif m != 0 and n == 0:
        return (
            (gamma1 - gamma2)
            * (b / a)
            * (-1)**abs(m)
            / (np.pi * m)
            * np.sin(m * np.pi * b / a)
        )
    elif m == 0 and n != 0:
        return (
            (gamma1 - gamma2)
            * (b / a)
            * (-1)**abs(n)
            / (np.pi * n)
            * np.sin(n * np.pi * b / a)
        )
    else:  # m != 0 and n != 0
        return (
            (gamma1 - gamma2)
            * (-1)**abs(m + n)
            / (np.pi**2 * m * n)
            * np.sin(m * np.pi * b / a)
            * np.sin(n * np.pi * b / a)
        )

def G_mn(m, n):
    return np.array([m * dG, n * dG])

'''max를 충족하는 m, n 리스트업'''
Gpoint_list = []

for m in range(-int(Gmf), int(Gmf) + 1):
    for n in range(-int(Gmf), int(Gmf) + 1):
        if m**2 + n**2 < Gmf**2:
            Gpoint_list.append((m, n))

# print(Gpoint_list[0])

'''k 받고 A, B 행렬 반환하는 함수 정의'''
def A_B_at_k(kvec):
    N = len(Gpoint_list)
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for i in range(len(Gpoint_list)):
        for j in range(len(Gpoint_list)):
            o, p = Gpoint_list[i]  # i번째 행에 대응하는 (o, p)
            m, n = Gpoint_list[j]  # j번째 열에 대응하는 (m, n)
            Gvec_i = G_mn(o, p)  # (o, p)에 대응하는 G벡터
            q_i=kvec + Gvec_i  # (o, p)에 대응하는 q벡터
            Gvec_j = G_mn(m, n)  # (m, n)에 대응하는 G벡터
            q_j=kvec + Gvec_j  # (m, n)에 대응하는 q벡터
            Gamma_ij = gamma_mn(o - m, p - n)  # (o-m, p-n)에 대응하는 감마 성분
            A[i, j] = Gamma_ij*np.linalg.norm(q_i)*np.linalg.norm(q_j)  # A 행렬 성분 계산
            B[i, j] = Gamma_ij*np.dot(q_i, q_j)
    return A, B

# A, B = A_B_at_k(np.array([0.0, 0.0]))

# print(A.shape)              # (697, 697)
# print(B.shape)              # (697, 697)
# print(np.allclose(A, A.T))  # True여야 함
# print(np.allclose(B, B.T))  # True여야 함

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
    TM_omega.append(omega_TM) #지금 rad*THz 단위인데, 격자가 nm 스케일이라 xray로 나올 수 있어서 위 함수 리스케일링 하면 됨

# print(TE_omega)
# print(TE_omega.shape)  # (106, 697)
# print(TM_omega)
# print(TM_omega.shape)  # (106, 697)
# len(TE_omega)  # 106
# len(TM_omega)  # 106
# len(TE_omega[0])  # 697

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = (
    f"2dpho_a={a:g}_d={d:g}_eps1={eps1:g}_eps2={eps2:g}_"
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

