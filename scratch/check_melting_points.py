import numpy as np

def einstein_G(T, E0, w, ThetaE):
    R = 8.314
    T = max(T, 1e-10)
    zpe = 0.5 * w * R * ThetaE
    vib = w * R * T * np.log(1 - np.exp(-ThetaE/T))
    return E0 + zpe + vib

def GLIQCU(T):
    if T < 1357.77:
        return 5194.277 + 120.97333*T - 24.112392*T*np.log(T) - 0.00265684*T**2 + 52478/T + 1.29223e-7*T**3 - 5.8489e-21*T**7
    else:
        return -46.545 + 173.881484*T - 31.38*T*np.log(T)

def GLIQAL(T):
    if T < 700:
        ghser = -7976.15 + 137.093038*T - 24.3671976*T*np.log(T) - 1.884662e-3*T**2 - 0.877664e-6*T**3 + 74092/T
    elif T < 933.473:
        ghser = -11276.24 + 223.048446*T - 38.5844296*T*np.log(T) + 18.531982e-3*T**2 - 5.764227e-6*T**3 + 74092/T
    else:
        ghser = -11278.361 + 188.684136*T - 31.748192*T*np.log(T) - 1230.622e25*T**(-9)
        
    if T < 933.473:
        return 11005.045 - 11.84185*T + 79.337e-21*T**7 + ghser
    else:
        return -795.991 + 177.430209*T - 31.748192*T*np.log(T)

T_vals = np.linspace(300, 2000, 100)
for T in T_vals:
    g_s = einstein_G(T, -7770, 3, 244)
    g_l = GLIQCU(T)
    if abs(g_s - g_l) < 1000:
        print(f"Cu intersection near T={T:.1f}: Solid={g_s:.1f}, Liq={g_l:.1f}, Diff={g_s-g_l:.1f}")

print("--- AL ---")
for T in T_vals:
    g_s = einstein_G(T, -7976, 3, 320)
    g_l = GLIQAL(T)
    if abs(g_s - g_l) < 1000:
        print(f"Al intersection near T={T:.1f}: Solid={g_s:.1f}, Liq={g_l:.1f}, Diff={g_s-g_l:.1f}")

