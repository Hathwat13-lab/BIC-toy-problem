'''DOS 없는 구현'''

import numpy as np
from datetime import datetime
from parameters import a, d, eps1, eps2, Gmf, Nx, c

# Shared parameters are defined in parameters.py.
b = a - d
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

# print(K)
# print(K.shape)  # (3*Nx + 1, 2) +1 은 감마 포인트가 중복으로 세지는 점을 고려해서...


'''감마, G 행렬 성분 정의 (벡터화 버전)

원래는 A_B_at_k 안에서 (i, j) 쌍마다 파이썬 이중 for문으로 gamma_mn을 호출했는데,
gamma_mn(o-m, p-n)은 kvec에 의존하지 않으므로 k마다 다시 계산할 필요가 없다.
Gmf=15 -> 40으로 늘리면 N(=Gpoint_list 길이)이 697 -> 5013으로 커져서 이중 루프가
k점당 90초 이상 걸려 전체 3시간 가까이 걸렸음. NxN Gamma 행렬을 최초 1회만 만들어
두고, k마다는 q벡터 갱신 + 행렬곱만 하도록 바꿔서 k점당 10초 수준으로 줄였다
(eigvalsh 자체가 N^3라 이 이상은 Gmf를 줄이거나 부분 고유값 솔버를 써야 함).
'''

def gamma_matrix(delta_m, delta_n):
    is_m0 = delta_m == 0
    is_n0 = delta_n == 0
    safe_m = np.where(is_m0, 1, delta_m)  # 0으로 나누기 방지용(해당 위치 결과는 안 씀)
    safe_n = np.where(is_n0, 1, delta_n)
    sign_m = np.where(delta_m % 2 == 0, 1.0, -1.0)   # (-1)**abs(m)과 동일
    sign_n = np.where(delta_n % 2 == 0, 1.0, -1.0)
    sign_mn = np.where((delta_m + delta_n) % 2 == 0, 1.0, -1.0)

    val00 = gamma2 + (gamma1 - gamma2) * (b / a)**2
    valM0 = (gamma1 - gamma2) * (b / a) * sign_m / (np.pi * safe_m) * np.sin(delta_m * np.pi * b / a)
    val0N = (gamma1 - gamma2) * (b / a) * sign_n / (np.pi * safe_n) * np.sin(delta_n * np.pi * b / a)
    valMN = (
        (gamma1 - gamma2) * sign_mn / (np.pi**2 * safe_m * safe_n)
        * np.sin(delta_m * np.pi * b / a) * np.sin(delta_n * np.pi * b / a)
    )
    return np.where(is_m0 & is_n0, val00, np.where(is_m0, val0N, np.where(is_n0, valM0, valMN)))

'''max를 충족하는 m, n 리스트업'''
Gpoint_list = []

for m in range(-int(Gmf), int(Gmf) + 1):
    for n in range(-int(Gmf), int(Gmf) + 1):
        if m**2 + n**2 < Gmf**2:
            Gpoint_list.append((m, n))

# print(Gpoint_list[0])

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

# A, B = A_B_at_k(np.array([0.0, 0.0]))

'''--- 기존 순수 파이썬 이중 루프 구현 (참고용, Gmf=40 기준 위 벡터화 버전보다 ~370배 느림) ---
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
--- 여기까지 참고용 ---'''

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
    TM_omega.append(omega_TM) 

#지금 rad*THz 단위인데, 격자가 nm 스케일이라 xray로 나올 수 있어서 위 함수 리스케일링 하면 됨

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

