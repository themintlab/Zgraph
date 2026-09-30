import numpy as np

def einstein_thermal(T, w, ThetaE):
    R = 8.314
    zpe = 0.5 * w * R * ThetaE
    vib = w * R * T * np.log(1 - np.exp(-ThetaE/T))
    return zpe + vib

def GLIQCU(T):
    if T < 1357.77:
        return 5194.277 + 120.97333*T - 24.112392*T*np.log(T) - 0.00265684*T**2 + 52478/T + 1.29223e-7*T**3 - 5.8489e-21*T**7
    else:
        return -46.545 + 173.881484*T - 31.38*T*np.log(T)

def GLIQAL(T):
    ghser = -11278.361 + 188.684136*T - 31.748192*T*np.log(T) - 1230.622e25*T**(-9)
    return -795.991 + 177.430209*T - 31.748192*T*np.log(T)

T_m_cu = 1357.77
G_liq_cu = GLIQCU(T_m_cu)
G_th_cu = einstein_thermal(T_m_cu, 3, 244)
E0_cu = G_liq_cu - G_th_cu

T_m_al = 933.473
G_liq_al = GLIQAL(T_m_al)
G_th_al = einstein_thermal(T_m_al, 3, 320)
E0_al = G_liq_al - G_th_al

print(f"Calibrated E0_Cu = {E0_cu:.1f}")
print(f"Calibrated E0_Al = {E0_al:.1f}")
