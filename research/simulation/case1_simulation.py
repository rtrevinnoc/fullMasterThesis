"""
Case 1 — Planar-Restricted Step-and-Scan Without Fine Stages
=============================================================
Based on:
  Al-Rawashdeh, Y.M., Al Janaideh, M., & Heertjes, M. (2022).
  "On characterization of a generic lithography machine in a multi-directional
  space." Mechanism and Machine Theory, 170, 104638.

Section 6.1 "Case 1":
  The machine without any fine stages, controlled by PID only.
  Planar restriction: motion in {x, y} only.

Machine structure (Appendix B)
───────────────────────────────
  Each chain is a 2-body mass–spring–damper (Kelvin–Voigt joints):
    • Coarse stage (m_c = m₃+m₅ = 77 kg):  actuated body, encoder mounted here.
    • End-effector (m_e = m₇+m₈):           payload, passively spring-coupled.

  Chains:  j=3  wafer   (x step, y scan)
           j=1  reticle (y scan only, opposite direction, 4× amplitude)

Control (collocated)
─────────────────────
  Feedback from coarse stage encoder.  Non-collocated control is unstable for
  the reticle chain (C_R = 710 N·s/m, ζ_ee = 0.04 → resonance excitation).

  F = Kp·(r − y_c) + Ki·∫(r − y_c) + Kd·(ṙ − ẏ_c) + Kff·r̈

  The synchronisation error  e_syn = (r_w − y_we) − α·(r_r − y_re)
  is non-zero because the spring coupling creates different lags in the wafer
  and reticle chains (different K, C, m_e).

  Static spring lag during acceleration a:  Δ = m_e·a / K
  Wafer lag:   Δ_w = 10.7 · 20  / 1.343 × 10⁷ ≈ 16 nm
  Reticle lag: Δ_r = 10.54 · 80 / 6.317 × 10⁶ ≈ 134 nm
  Quasi-static e_syn ≈ Δ_w − α·Δ_r ≈ 16 − 0.25·134 ≈ −17 nm (during ramp)
  + residual oscillation of lightly-damped reticle end-effector at 123 Hz.

Performance indices (Eqs. 5.1–5.4)
────────────────────────────────────
  e_syn = e_w,y − α · e_r,y
  MA spec:  e_syn ∈ [−1.25, +2] nm
  MSD spec: MSD ≤ 7 nm
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════════
# 1. MACHINE PARAMETERS  (Appendix B, translational DOF)
# ═══════════════════════════════════════════════════════════════════════════

ALPHA = 0.25    # demagnification: reticle moves 4× faster in opposite direction

# --- Wafer chain (j = 3) ---
M_WC = 49.0 + 28.0        # kg   m₃ + m₅
M_WE = 10.5  + 0.2         # kg   m₇ + m₈
K_W  = 1e9 * 0.01343       # N/m  K₇ translational ≈ 1.343 × 10⁷ N/m
C_W  = 1e8 * 0.0001679     # N·s/m C₇ ≈ 1.679 × 10⁴ N·s/m
# End-effector natural freq & damping (open loop):
ω_WE = np.sqrt(K_W * (M_WC + M_WE) / (M_WC * M_WE))  # 2-mass mode
ζ_WE = C_W / (2.0 * np.sqrt(K_W * M_WE))

# --- Reticle chain (j = 1) – no fine stage ---
M_RC = 49.0 + 28.0        # kg
M_RE = 10.5  + 0.04        # kg   (lighter reticle mask)
K_R  = 1e9 * 0.0063165    # N/m  K₈ reticle ≈ 6.317 × 10⁶ N/m
C_R  = 710.0              # N·s/m C₈ reticle (Appendix B.1.1) — lightly damped!
ω_RE = np.sqrt(K_R * (M_RC + M_RE) / (M_RC * M_RE))
ζ_RE = C_R / (2.0 * np.sqrt(K_R * M_RE))

# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAJECTORY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

V_W_SCAN   = 0.8              # m/s  wafer scan speed
V_R_SCAN   = V_W_SCAN/ALPHA   # 3.2 m/s  reticle scan speed
T_EXP      = 5.5e-3 / V_R_SCAN   # 1.71875 ms  exposure time (Eq. 5.1)

DIE_Y      = 32e-3;  DIE_X = 32e-3;  SCRIBE = 0.2e-3
STEP_PITCH = DIE_X + SCRIBE     # 32.2 mm

N_DIES     = 4          # four 32×32 mm² dies (1 row × 4 columns, snake pattern)

# Scan profile (wafer scan axis)
A_W_MAX   = 20.0        # m/s²
J_W_MAX   = 1600.0      # m/s³  (T₁ = 12.5 ms)

# Step profile
A_X_MAX   = 45.0        # m/s²  (≤ 45 m/s² from Butler [2])
J_X_MAX   = 3600.0      # m/s³

# Reticle scan (4× speed)
A_R_MAX   = A_W_MAX / ALPHA     # 80 m/s²
J_R_MAX   = J_W_MAX / ALPHA     # 6400 m/s³

T_SETTLE  = 0.15        # s  settling time between scan and step (150 ms → total ~1.73 s)

DT_REF    = 1e-5        # reference sample period (10 µs)

# ═══════════════════════════════════════════════════════════════════════════
# 3. COLLOCATED PID + INERTIA FEEDFORWARD
#    Feedback on coarse stage position — always stable for this plant.
#    Bandwidth: ωc = 2π × 50 rad/s
#    Kp = M_total · ωc²,  Kd = M_total · ωc,  Ki = Kp · ωc / 10
# ═══════════════════════════════════════════════════════════════════════════

OMC     = 2.0 * np.pi * 50.0    # 314.2 rad/s

M_W_TOT = M_WC + M_WE           # 87.7 kg
M_R_TOT = M_RC + M_RE           # 87.54 kg

KP_W = M_W_TOT * OMC**2         # 8.656 × 10⁶ N/m
KD_W = M_W_TOT * OMC            # 2.754 × 10⁴ N·s/m
KI_W = KP_W * OMC / 10.0        # 2.720 × 10⁸ N/(m·s)
FF_W = M_W_TOT                  # inertia feedforward [kg]

KP_R = M_R_TOT * OMC**2         # 8.641 × 10⁶ N/m
KD_R = M_R_TOT * OMC            # 2.749 × 10⁴ N·s/m
KI_R = KP_R * OMC / 10.0        # 2.716 × 10⁸ N/(m·s)
FF_R = M_R_TOT                  # inertia feedforward

F_SAT = 5e4    # N  control force saturation

print("═" * 64)
print("Case 1 — Planar Step-and-Scan, No Fine Stages")
print("═" * 64)
print(f"  Wafer chain :  coarse {M_WC:.0f} kg + end-eff {M_WE:.1f} kg")
print(f"    K_W = {K_W:.2e} N/m,  C_W = {C_W:.2e} N·s/m")
print(f"    2-mass resonance = {ω_WE/(2*np.pi):.1f} Hz,  ζ_ee = {ζ_WE:.3f}")
print(f"  Reticle chain: coarse {M_RC:.0f} kg + end-eff {M_RE:.2f} kg")
print(f"    K_R = {K_R:.2e} N/m,  C_R = {C_R:.1f} N·s/m")
print(f"    2-mass resonance = {ω_RE/(2*np.pi):.1f} Hz,  ζ_ee = {ζ_RE:.3f} (lightly damped)")
print(f"  Control: collocated PID+FF, ωc = {OMC/(2*np.pi):.1f} Hz")
print(f"  Scan: wafer {V_W_SCAN*1e3:.0f} mm/s | reticle {V_R_SCAN*1e3:.0f} mm/s (opposite)")
print(f"  T_exp = {T_EXP*1e6:.2f} µs per die")

# ═══════════════════════════════════════════════════════════════════════════
# 4. 3RD-ORDER MOTION PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def make_profile(v_peak, a_max, j_max, d_cruise, dt=DT_REF):
    """
    Symmetric 3rd-order (constant-jerk) profile: rest → v_peak → rest.
    Returns t, pos, vel, acc, jrk, t_cruise_start, t_cruise_end.
    """
    T1 = a_max / j_max
    v1 = 0.5 * j_max * T1**2
    vr = v_peak - 2.0 * v1
    if vr < -1e-12:
        T1 = np.sqrt(v_peak / j_max);  T2 = 0.0
        a_max = j_max * T1;            v1 = 0.5 * j_max * T1**2;  vr = 0.0
    else:
        T2 = vr / a_max

    p1 = j_max*T1**3/6
    v2 = v1+a_max*T2;          p2 = p1+v1*T2+0.5*a_max*T2**2
    v3 = v2+a_max*T1-0.5*j_max*T1**2
    p3 = p2+v2*T1+0.5*a_max*T1**2-j_max*T1**3/6
    p4 = p3+v3*d_cruise
    v5 = v3-0.5*j_max*T1**2;  p5 = p4+v3*T1-j_max*T1**3/6
    v6 = v5-a_max*T2;          p6 = p5+v5*T2-0.5*a_max*T2**2
    p7 = p6+v6*T1-0.5*a_max*T1**2+j_max*T1**3/6

    tb = [0, T1, T1+T2, 2*T1+T2,
          2*T1+T2+d_cruise, 3*T1+T2+d_cruise,
          3*T1+2*T2+d_cruise, 4*T1+2*T2+d_cruise]
    jv  = [j_max, 0, -j_max, 0, -j_max, 0, j_max]
    s0  = [(0,0,0),(a_max,v1,p1),(a_max,v2,p2),(0,v3,p3),
           (0,v3,p4),(-a_max,v5,p5),(-a_max,v6,p6)]

    t   = np.arange(0.0, tb[-1]+dt/2, dt)
    pos = np.zeros_like(t);  vel = np.zeros_like(t)
    acc = np.zeros_like(t);  jrk = np.zeros_like(t)

    for ph in range(7):
        ts, te = tb[ph], tb[ph+1]
        j, (a0,v0,p0) = jv[ph], s0[ph]
        if ph < 6:
            m = (t >= ts-dt/4) & (t < te+dt/4)
        else:
            m = (t >= ts-dt/4)
        tau     = t[m] - ts
        jrk[m]  = j
        acc[m]  = a0 + j*tau
        vel[m]  = v0 + a0*tau + 0.5*j*tau**2
        pos[m]  = p0 + v0*tau + 0.5*a0*tau**2 + (1.0/6.0)*j*tau**3

    return t, pos, vel, acc, jrk, tb[3], tb[4]


def step_dcruise(target, a_max, j_max):
    _, sp, *_ = make_profile(a_max**2/j_max, a_max, j_max, 0.0, dt=1e-4)
    return max(0.0, (target - sp[-1]) / (a_max**2/j_max))

D_CRUISE_Y = DIE_Y / V_W_SCAN          # 40 ms — wafer scans one die at v_w
D_CRUISE_X = step_dcruise(STEP_PITCH, A_X_MAX, J_X_MAX)

sc_t, sc_p, sc_v, sc_a, sc_j, T_SC_CS, T_SC_CE = \
    make_profile(V_W_SCAN, A_W_MAX, J_W_MAX, D_CRUISE_Y)
st_t, st_p, st_v, st_a, st_j, _, _ = \
    make_profile(A_X_MAX**2/J_X_MAX, A_X_MAX, J_X_MAX, D_CRUISE_X)

T_SCAN = sc_t[-1];  T_STEP = st_t[-1]
T_EXP_CENTRE = (T_SC_CS + T_SC_CE) / 2.0

print(f"\n  Scan profile: {T_SCAN*1e3:.1f} ms  "
      f"cruise [{T_SC_CS*1e3:.1f}, {T_SC_CE*1e3:.1f}] ms")
print(f"  Step profile: {T_STEP*1e3:.1f} ms")

# ═══════════════════════════════════════════════════════════════════════════
# 5. BUILD FULL STEP-AND-SCAN REFERENCE TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════

def build_trajectories():
    """Build wafer (x,y) and reticle (y) reference arrays for all N_DIES."""
    keys = ['t', 'yw', 'yv', 'ya', 'yr', 'rv', 'ra', 'xw', 'xv', 'xa']
    segs = {k: [] for k in keys}
    exp_windows = []
    t_g = 0.0;  yw = 0.0;  yr = 0.0;  xw = 0.0

    n_idle = max(1, int(T_SETTLE / DT_REF))
    ti     = np.arange(n_idle) * DT_REF
    z      = np.zeros(n_idle)

    for d in range(N_DIES):
        dr = 1 if (d % 2 == 0) else -1   # bidirectional scan

        # scan segment
        segs['t'].append(sc_t + t_g)
        segs['yw'].append(yw + dr*sc_p);  segs['yv'].append(dr*sc_v);  segs['ya'].append(dr*sc_a)
        segs['yr'].append(yr - dr*sc_p/ALPHA); segs['rv'].append(-dr*sc_v/ALPHA); segs['ra'].append(-dr*sc_a/ALPHA)
        segs['xw'].append(np.full(len(sc_t), xw)); segs['xv'].append(np.zeros(len(sc_t))); segs['xa'].append(np.zeros(len(sc_t)))
        exp_windows.append((t_g + T_EXP_CENTRE - T_EXP/2,
                            t_g + T_EXP_CENTRE + T_EXP/2))
        yw += dr*sc_p[-1];  yr -= dr*sc_p[-1]/ALPHA;  t_g += T_SCAN

        if d < N_DIES - 1:
            # idle settle
            segs['t'].append(ti+t_g); segs['yw'].append(np.full(n_idle,yw)); segs['yv'].append(z); segs['ya'].append(z)
            segs['yr'].append(np.full(n_idle,yr)); segs['rv'].append(z); segs['ra'].append(z)
            segs['xw'].append(np.full(n_idle,xw)); segs['xv'].append(z); segs['xa'].append(z)
            t_g += T_SETTLE
            # x step
            segs['t'].append(st_t+t_g); segs['yw'].append(np.full(len(st_t),yw)); segs['yv'].append(np.zeros(len(st_t))); segs['ya'].append(np.zeros(len(st_t)))
            segs['yr'].append(np.full(len(st_t),yr)); segs['rv'].append(np.zeros(len(st_t))); segs['ra'].append(np.zeros(len(st_t)))
            segs['xw'].append(xw+st_p); segs['xv'].append(st_v); segs['xa'].append(st_a)
            xw += st_p[-1];  t_g += T_STEP
            # idle again
            segs['t'].append(ti+t_g); segs['yw'].append(np.full(n_idle,yw)); segs['yv'].append(z); segs['ya'].append(z)
            segs['yr'].append(np.full(n_idle,yr)); segs['rv'].append(z); segs['ra'].append(z)
            segs['xw'].append(np.full(n_idle,xw)); segs['xv'].append(z); segs['xa'].append(z)
            t_g += T_SETTLE

    return {k: np.concatenate(v) for k, v in segs.items()}, exp_windows


print("\nBuilding reference trajectories …")
ref, exp_windows = build_trajectories()
t_ref = ref['t'];  T_SIM = t_ref[-1]
print(f"  T_sim = {T_SIM*1e3:.1f} ms,  {len(t_ref)} reference points")
print(f"  {len(exp_windows)} exposure windows of {T_EXP*1e6:.1f} µs each")

# ═══════════════════════════════════════════════════════════════════════════
# 6. CLOSED-LOOP ODE  (15 states)
# ═══════════════════════════════════════════════════════════════════════════
#
# State:
#  [0,1]  y_wc, ẏ_wc    wafer coarse y
#  [2,3]  y_we, ẏ_we    wafer end-effector y
#  [4,5]  x_wc, ẋ_wc    wafer coarse x
#  [6,7]  x_we, ẋ_we    wafer end-effector x
#  [8,9]  y_rc, ẏ_rc    reticle coarse y
# [10,11] y_re, ẏ_re    reticle end-effector y
# [12]    I_wy           integral – wafer y (coarse error)
# [13]    I_wx           integral – wafer x
# [14]    I_ry           integral – reticle y
#
# 2-mass dynamics:
#   M_c · ÿ_c = F − K(y_c−y_e) − C(ẏ_c−ẏ_e)
#   M_e · ÿ_e =     K(y_c−y_e) + C(ẏ_c−ẏ_e)
#
# Collocated control (coarse stage feedback):
#   e_c = r(t) − y_c
#   F   = Kp·e_c + Ki·I + Kd·(ṙ − ẏ_c) + Kff·r̈

def _ip(t, arr):
    return float(np.interp(t, t_ref, arr))

def ode(t, s):
    ywc,dywc,ywe,dywe, xwc,dxwc,xwe,dxwe, yrc,dyrc,yre,dyre, Iwy,Iwx,Iry = s

    ryw = _ip(t, ref['yw']); vyw = _ip(t, ref['yv']); ayw = _ip(t, ref['ya'])
    rxw = _ip(t, ref['xw']); vxw = _ip(t, ref['xv']); axw = _ip(t, ref['xa'])
    ryr = _ip(t, ref['yr']); vyr = _ip(t, ref['rv']); ayr = _ip(t, ref['ra'])

    # Wafer y  (collocated: error on coarse stage)
    ec_wy = ryw - ywc
    F_wy  = np.clip(KP_W*ec_wy + KI_W*Iwy + KD_W*(vyw-dywc) + FF_W*ayw, -F_SAT, F_SAT)
    ddywc = (F_wy - K_W*(ywc-ywe) - C_W*(dywc-dywe)) / M_WC
    ddywe = (       K_W*(ywc-ywe) + C_W*(dywc-dywe)) / M_WE

    # Wafer x
    ec_wx = rxw - xwc
    F_wx  = np.clip(KP_W*ec_wx + KI_W*Iwx + KD_W*(vxw-dxwc) + FF_W*axw, -F_SAT, F_SAT)
    ddxwc = (F_wx - K_W*(xwc-xwe) - C_W*(dxwc-dxwe)) / M_WC
    ddxwe = (       K_W*(xwc-xwe) + C_W*(dxwc-dxwe)) / M_WE

    # Reticle y  (collocated: error on coarse stage)
    ec_ry = ryr - yrc
    F_ry  = np.clip(KP_R*ec_ry + KI_R*Iry + KD_R*(vyr-dyrc) + FF_R*ayr, -F_SAT, F_SAT)
    ddyrc = (F_ry - K_R*(yrc-yre) - C_R*(dyrc-dyre)) / M_RC
    ddyre = (       K_R*(yrc-yre) + C_R*(dyrc-dyre)) / M_RE

    return [dywc,ddywc,dywe,ddywe, dxwc,ddxwc,dxwe,ddxwe,
            dyrc,ddyrc,dyre,ddyre, ec_wy,ec_wx,ec_ry]


# ═══════════════════════════════════════════════════════════════════════════
# 7. INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

print("\nIntegrating ODE …")
sol = solve_ivp(ode, [0.0, T_SIM], [0.0]*15,
                method='RK45', max_step=1e-4,
                rtol=1e-6, atol=1e-9,
                dense_output=False)
print(f"  Steps: {len(sol.t)}   Success: {sol.success}")

t_s   = sol.t
y_wc  = sol.y[0];  dy_wc = sol.y[1]
y_we  = sol.y[2];  dy_we = sol.y[3]
x_wc  = sol.y[4];  dx_wc = sol.y[5]
x_we  = sol.y[6];  dx_we = sol.y[7]
y_rc  = sol.y[8];  dy_rc = sol.y[9]
y_re  = sol.y[10]; dy_re = sol.y[11]

# References at solver time points
yw_rs = np.interp(t_s, t_ref, ref['yw'])
xw_rs = np.interp(t_s, t_ref, ref['xw'])
yr_rs = np.interp(t_s, t_ref, ref['yr'])

# Tracking errors at END-EFFECTORS [nm] (what matters for lithography)
e_wy  = (yw_rs - y_we) * 1e9
e_wx  = (xw_rs - x_we) * 1e9
e_ry  = (yr_rs - y_re) * 1e9
# Coarse-stage errors (for reference)
ec_wy = (yw_rs - y_wc) * 1e9
ec_ry = (yr_rs - y_rc) * 1e9

# Synchronisation error  Eq.(5.2)
e_syn = e_wy - ALPHA * e_ry

# ═══════════════════════════════════════════════════════════════════════════
# 8. PERFORMANCE INDICES
# ═══════════════════════════════════════════════════════════════════════════

def ma_msd(t_arr, sig, window):
    """Vectorised moving average and MSD via uniform resampling + cumsum."""
    dt_u  = np.median(np.diff(t_arr))
    t_u   = np.arange(t_arr[0], t_arr[-1], dt_u)
    s_u   = np.interp(t_u, t_arr, sig)
    hw    = max(1, int(round(0.5*window/dt_u)))
    pad   = np.pad(s_u, hw, mode='edge')
    cs    = np.cumsum(pad);   cs2 = np.cumsum(pad**2)
    n     = 2*hw+1
    ma_u  = (cs[2*hw:] - cs[:len(s_u)]) / n
    m2_u  = (cs2[2*hw:] - cs2[:len(s_u)]) / n
    msd_u = np.sqrt(np.maximum(m2_u - ma_u**2, 0.0))
    return np.interp(t_arr, t_u, ma_u), np.interp(t_arr, t_u, msd_u)


print("Computing MA / MSD …")
ma_syn, msd_syn = ma_msd(t_s, e_syn, T_EXP)

in_exp = np.zeros(len(t_s), dtype=bool)
for t0, t1 in exp_windows:
    in_exp |= (t_s >= t0) & (t_s <= t1)

ma_exp  = ma_syn[in_exp]
msd_exp = msd_syn[in_exp]

print(f"  MA_syn  during exposure : [{ma_exp.min():+.2f}, {ma_exp.max():+.2f}] nm")
print(f"  MSD_syn during exposure :  max = {msd_exp.max():.2f} nm")
print(f"  Spec: MA ∈ [−1.25, +2] nm,  MSD ≤ 7 nm  (38 nm half-pitch)")

t_s_sec = t_s          # time in seconds (for paper-style plots)
t_ms    = t_s * 1e3   # time in ms (for diagnostic plots)

# ═══════════════════════════════════════════════════════════════════════════
# 9. PLOTS — matching paper's figures
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 12 (paper) — Desired step-and-scan trajectories ─────────────────
#   Top:    Wafer Y reference (scanning direction) vs time [seconds]
#   Bottom: Wafer X reference (stepping direction) vs time [seconds]
fig1, (ax12a, ax12b) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

ax12a.plot(t_s_sec, yw_rs, color='#1f77b4', lw=1.0, label='Desired Scanning Position')
ax12a.set_ylabel('Y Position (m)')
ax12a.legend(fontsize=9, loc='upper right')
ax12a.grid(True, alpha=0.25)
ax12a.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

ax12b.plot(t_s_sec, xw_rs, color='#1f77b4', lw=1.0, label='Desired Stepping Position')
ax12b.set_ylabel('X Position (m)')
ax12b.set_xlabel('Time (second)')
ax12b.legend(fontsize=9, loc='upper left')
ax12b.grid(True, alpha=0.25)
ax12b.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

fig1.suptitle('Fig. 12 — Desired Step-and-Scan Trajectories\n'
              '(relative to lens element coordinates, inspired by Butler [2])',
              fontsize=10)
fig1.tight_layout()
fig1.savefig('case1_fig12_trajectories.png', dpi=150, bbox_inches='tight')
print('\n  Saved: case1_fig12_trajectories.png')

# ── Figure 2 (diagnostic): Tracking errors ────────────────────────────────
fig2 = plt.figure(figsize=(13, 10))
gs   = GridSpec(4, 1, figure=fig2, hspace=0.5)
axes = [fig2.add_subplot(gs[i]) for i in range(4)]

for ax_t, err, lbl, col in [
    (axes[0], e_wy,  r'Oblea $e_{w,y}$ (nm)',     '#1f77b4'),
    (axes[1], e_ry,  r'Retícula $e_{r,y}$ (nm)',  '#ff7f0e'),
    (axes[2], e_wx,  r'Oblea $e_{w,x}$ (nm)',     '#2ca02c'),
    (axes[3], e_syn, r'Sinc. $e_{syn}$ (nm)',     '#d62728'),
]:
    ax_t.plot(t_s_sec, err, color=col, lw=0.6)
    ax_t.axhline(0, color='k', lw=0.5, ls='--')
    ax_t.set_ylabel(lbl, fontsize=11);  ax_t.grid(True, alpha=0.25)
    ax_t.tick_params(labelsize=10)
    for t0, t1 in exp_windows:
        ax_t.axvspan(t0, t1, alpha=0.18, color='purple')

axes[3].axhline( 1.0,  color='k', ls=':', lw=1.0, label='Espec. MA ±1 nm')
axes[3].axhline(-1.0,  color='k', ls=':', lw=1.0)
axes[3].legend(fontsize=10);  axes[3].set_xlabel('Tiempo (s)', fontsize=11)
fig2.suptitle('Error de seguimiento (sin etapa fina)\n'
              'Bandas sombreadas: ventanas de exposición', fontsize=13, fontweight='bold')
fig2.savefig('case1_tracking_errors.png', dpi=150, bbox_inches='tight')
print('  Saved: case1_tracking_errors.png')

# ── Figure 14 (paper) — MA and MSD performance indices ─────────────────────
#   Matches paper's Fig. 14: time in seconds, y-axis in metres, spec lines,
#   inset zoom around last exposure window (×10⁻⁸ scale).
MA_UPPER =  2.0e-9    # +2 nm spec
MA_LOWER = -1.25e-9   # -1.25 nm spec
MSD_UPPER = 7.0e-9    # 7 nm spec

# Convert signals to metres for paper-matching axes
ma_m   = ma_syn  * 1e-9
msd_m  = msd_syn * 1e-9

fig3, (axma, axmsd) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Clip data at axis limits so out-of-range values appear as flat lines at the boundary
# (without clipping, matplotlib makes those segments invisible, hiding the saturation)
MA_LIM   = 1e-6
MSD_LIM  = 2.5e-8
ma_m_clipped  = np.clip(ma_m,  -MA_LIM,  MA_LIM)
msd_m_clipped = np.clip(msd_m, 0.0,      MSD_LIM)

# --- MA panel (a) ---
axma.plot(t_s_sec, ma_m_clipped, color='#1f4e79', lw=0.7, label='MA (sin etapa fina)')
axma.axhline(MA_UPPER, color='#c00000', ls='--', lw=1.0, label='Cota sup. MA')
axma.axhline(MA_LOWER, color='#843c0c', ls='--', lw=1.0, label='Cota inf. MA')
axma.set_ylabel('Promedio móvil (m)', fontsize=11)
axma.yaxis.set_major_formatter(
    plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
axma.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axma.set_ylim(-1e-6, 1e-6)   # clip like paper: No Fine Stage saturates; shows fine-stage range
axma.legend(fontsize=10, loc='upper right')
axma.tick_params(labelsize=10)
axma.grid(True, alpha=0.2)
axma.text(0.01, 0.97, '(a)', transform=axma.transAxes,
          va='top', fontsize=11, fontweight='bold')

# Inset: last cruise window at ×10⁻⁸ scale (matching paper's inset style)
tw_last = exp_windows[-1]
t_ins_l = max(0, tw_last[0] - 0.08)
t_ins_r = min(T_SIM, tw_last[1] + 0.08)
m_ins = (t_s_sec >= t_ins_l) & (t_s_sec <= t_ins_r)
ax_ins = axma.inset_axes([0.55, 0.10, 0.43, 0.58])
ax_ins.plot(t_s_sec[m_ins], ma_m[m_ins], color='#1f4e79', lw=0.8)  # unclipped for inset detail
ax_ins.axhline(MA_UPPER, color='#c00000', ls='--', lw=0.8)
ax_ins.axhline(MA_LOWER, color='#843c0c', ls='--', lw=0.8)
for t0, t1 in exp_windows:
    if t_ins_l <= t0 <= t_ins_r:
        ax_ins.axvspan(t0, t1, alpha=0.25, color='purple')
ax_ins.ticklabel_format(axis='y', style='sci', scilimits=(-8,-8))
ax_ins.tick_params(labelsize=6)
ax_ins.set_xlim(t_ins_l, t_ins_r)
ax_ins.set_ylim(-2e-6, 2e-6)
axma.indicate_inset_zoom(ax_ins, edgecolor='gray', lw=0.8)

# --- MSD panel (b) ---
axmsd.plot(t_s_sec, msd_m_clipped, color='#1f4e79', lw=0.7, label='MSD (sin etapa fina)')
axmsd.axhline(MSD_UPPER, color='#c00000', ls='--', lw=1.0, label='Cota sup. MSD')
axmsd.set_ylabel('Desviación estándar móvil (m)', fontsize=11)
axmsd.set_xlabel('Tiempo (s)', fontsize=11)
axmsd.yaxis.set_major_formatter(
    plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
axmsd.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axmsd.set_ylim(0, 2.5e-8)   # clip like paper: shows fine-stage spec range; No Fine Stage is off-scale
axmsd.legend(fontsize=10, loc='upper right')
axmsd.tick_params(labelsize=10)
axmsd.grid(True, alpha=0.2)
axmsd.text(0.01, 0.97, '(b)', transform=axmsd.transAxes,
           va='top', fontsize=11, fontweight='bold')

# Inset for MSD – same time window
ax_ins2 = axmsd.inset_axes([0.55, 0.35, 0.43, 0.58])
ax_ins2.plot(t_s_sec[m_ins], msd_m[m_ins], color='#1f4e79', lw=0.8)
ax_ins2.axhline(MSD_UPPER, color='#c00000', ls='--', lw=0.8)
for t0, t1 in exp_windows:
    if t_ins_l <= t0 <= t_ins_r:
        ax_ins2.axvspan(t0, t1, alpha=0.25, color='purple')
ax_ins2.ticklabel_format(axis='y', style='sci', scilimits=(-8,-8))
ax_ins2.tick_params(labelsize=6)
ax_ins2.set_xlim(t_ins_l, t_ins_r)
axmsd.indicate_inset_zoom(ax_ins2, edgecolor='gray', lw=0.8)

fig3.suptitle('Índices de rendimiento MA y MSD\n'
              '(sin etapa fina · PID+FF colocado)',
              fontsize=13, fontweight='bold')
fig3.tight_layout()
fig3.savefig('case1_fig14_ma_msd.png', dpi=150, bbox_inches='tight')
print('  Saved: case1_fig14_ma_msd.png')

# ── Figure: Y-axis sync error + MA + MSD (presentation column) ──────────────
fig_y, axes_y = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Zoom window: 60 ms centred on first exposure
ew0_c = (exp_windows[0][0] + exp_windows[0][1]) / 2.0
z_lo, z_hi = ew0_c - 0.030, ew0_c + 0.030
zm = (t_s_sec >= z_lo) & (t_s_sec <= z_hi)

# ── panel (a): e_syn ──────────────────────────────────────────────────────────
axes_y[0].plot(t_s_sec, e_syn, color='#d62728', lw=0.6)
axes_y[0].axhline(0, color='k', lw=0.5, ls='--')
axes_y[0].set_ylabel(r'$e_{syn}$ (nm)', fontsize=10)
axes_y[0].set_ylim(-60, 60)
axes_y[0].grid(True, alpha=0.25)
for t0, t1 in exp_windows:
    axes_y[0].axvspan(t0, t1, alpha=0.55, color='gold', zorder=3)
axes_y[0].text(0.01, 0.95, '(a) Error $e_{syn}$', transform=axes_y[0].transAxes,
               va='top', fontsize=9, fontweight='bold')

ax0_ins = axes_y[0].inset_axes([0.55, 0.08, 0.43, 0.72])
ax0_ins.plot(t_s_sec[zm], e_syn[zm], color='#d62728', lw=0.8)
ax0_ins.axhline(0, color='k', lw=0.5, ls='--')
ax0_ins.axvspan(exp_windows[0][0], exp_windows[0][1], alpha=0.65, color='gold',
                label='Exposición', zorder=3)
ax0_ins.set_ylim(-60, 60)
ax0_ins.set_xlim(z_lo, z_hi)
ax0_ins.tick_params(labelsize=6)
ax0_ins.set_xlabel('t (s)', fontsize=6)
ax0_ins.legend(fontsize=6, loc='lower right')
axes_y[0].indicate_inset_zoom(ax0_ins, edgecolor='0.4', lw=0.7)

# ── panel (b): MA ─────────────────────────────────────────────────────────────
ma_clip = np.clip(ma_syn, -50, 50)
axes_y[1].plot(t_s_sec, ma_clip, color='#1f4e79', lw=0.7)
axes_y[1].axhline( 2.0,  color='#c00000', ls='--', lw=1.0, label='+2 nm')
axes_y[1].axhline(-1.25, color='#843c0c', ls='--', lw=1.0, label='-1.25 nm')
axes_y[1].set_ylabel('MA (nm)', fontsize=10)
axes_y[1].set_ylim(-10, 10)
axes_y[1].legend(fontsize=8, loc='upper right', ncol=2)
axes_y[1].grid(True, alpha=0.2)
for t0, t1 in exp_windows:
    axes_y[1].axvspan(t0, t1, alpha=0.55, color='gold', zorder=3)
axes_y[1].text(0.01, 0.95, '(b) MA', transform=axes_y[1].transAxes,
               va='top', fontsize=9, fontweight='bold')

ax1_ins = axes_y[1].inset_axes([0.55, 0.08, 0.43, 0.72])
ax1_ins.plot(t_s_sec[zm], np.clip(ma_syn[zm], -10, 10), color='#1f4e79', lw=0.8)
ax1_ins.axhline( 2.0,  color='#c00000', ls='--', lw=0.8)
ax1_ins.axhline(-1.25, color='#843c0c', ls='--', lw=0.8)
ax1_ins.axvspan(exp_windows[0][0], exp_windows[0][1], alpha=0.65, color='gold', zorder=3)
ax1_ins.set_xlim(z_lo, z_hi)
ax1_ins.set_ylim(-10, 10)
ax1_ins.tick_params(labelsize=6)
ax1_ins.set_xlabel('t (s)', fontsize=6)
axes_y[1].indicate_inset_zoom(ax1_ins, edgecolor='0.4', lw=0.7)

# ── panel (c): MSD ────────────────────────────────────────────────────────────
msd_clip = np.clip(msd_syn, 0, 30)
axes_y[2].plot(t_s_sec, msd_clip, color='#1f4e79', lw=0.7)
axes_y[2].axhline(7.0, color='#c00000', ls='--', lw=1.0, label='7 nm')
axes_y[2].set_ylabel('MSD (nm)', fontsize=10)
axes_y[2].set_ylim(0, 30)
axes_y[2].set_xlabel('Tiempo (s)', fontsize=10)
axes_y[2].legend(fontsize=8, loc='upper right')
axes_y[2].grid(True, alpha=0.2)
for t0, t1 in exp_windows:
    axes_y[2].axvspan(t0, t1, alpha=0.55, color='gold', zorder=3)
axes_y[2].text(0.01, 0.95, '(c) MSD', transform=axes_y[2].transAxes,
               va='top', fontsize=9, fontweight='bold')

ax2_ins = axes_y[2].inset_axes([0.55, 0.20, 0.43, 0.72])
ax2_ins.plot(t_s_sec[zm], msd_syn[zm], color='#1f4e79', lw=0.8)
ax2_ins.axhline(7.0, color='#c00000', ls='--', lw=0.8)
ax2_ins.axvspan(exp_windows[0][0], exp_windows[0][1], alpha=0.65, color='gold', zorder=3)
ax2_ins.set_xlim(z_lo, z_hi)
ax2_ins.set_ylim(0, 30)
ax2_ins.tick_params(labelsize=6)
ax2_ins.set_xlabel('t (s)', fontsize=6)
axes_y[2].indicate_inset_zoom(ax2_ins, edgecolor='0.4', lw=0.7)

fig_y.suptitle('Dirección Y — error, MA y MSD\n(PID LS)',
               fontsize=11, fontweight='bold')
fig_y.tight_layout()
_out_y = '../../presentation/pictures/sim_combined_y.png'
fig_y.savefig(_out_y, dpi=150, bbox_inches='tight')
fig_y.savefig('sim_combined_y.png', dpi=150, bbox_inches='tight')
print(f'  Saved: sim_combined_y.png → {_out_y}')

# ── Figure 4: Velocities + coarse/end-effector lag ─────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(13, 8))
fig4.suptitle('Case 1 — Stage Velocities and Coarse-vs-End-Effector Lag',
              fontsize=11, fontweight='bold')

axes4[0,0].plot(t_s_sec, dy_we, '#1f77b4', lw=0.8, label='Wafer y  end-eff.')
axes4[0,0].plot(t_s_sec, dy_wc, '#aec7e8', lw=0.5, alpha=0.8, label='Wafer y  coarse')
axes4[0,0].plot(t_s_sec, np.interp(t_s,t_ref,ref['yv']), 'k--', lw=0.6, alpha=0.5, label='Reference')
axes4[0,0].set_ylabel('v (m/s)');  axes4[0,0].set_title('Wafer scan (y) velocity')
axes4[0,0].legend(fontsize=7);  axes4[0,0].grid(True, alpha=0.25)

axes4[0,1].plot(t_s_sec, dx_we, '#2ca02c', lw=0.8, label='Wafer x  end-eff.')
axes4[0,1].plot(t_s_sec, dx_wc, '#98df8a', lw=0.5, alpha=0.8, label='Wafer x  coarse')
axes4[0,1].set_ylabel('v (m/s)');  axes4[0,1].set_title('Wafer step (x) velocity')
axes4[0,1].legend(fontsize=7);  axes4[0,1].grid(True, alpha=0.25)

axes4[1,0].plot(t_s_sec, dy_re, '#ff7f0e', lw=0.8, label='Reticle y  end-eff.')
axes4[1,0].plot(t_s_sec, dy_rc, '#ffbb78', lw=0.5, alpha=0.8, label='Reticle y  coarse')
axes4[1,0].plot(t_s_sec, np.interp(t_s,t_ref,ref['rv']), 'k--', lw=0.6, alpha=0.5, label='Reference')
axes4[1,0].set_ylabel('v (m/s)');  axes4[1,0].set_title('Reticle scan (y) velocity')
axes4[1,0].legend(fontsize=7);  axes4[1,0].grid(True, alpha=0.25)

# Spring lag (coarse - end-effector displacement)
lag_w = (y_wc - y_we)*1e9   # nm
lag_r = (y_rc - y_re)*1e9   # nm
axes4[1,1].plot(t_s_sec, lag_w, '#1f77b4', lw=0.9, label='Wafer spring lag (nm)')
axes4[1,1].plot(t_s_sec, lag_r, '#ff7f0e', lw=0.9, label='Reticle spring lag (nm)')
axes4[1,1].set_ylabel('Spring lag y_c − y_e (nm)');  axes4[1,1].set_title('Spring coupling lag')
axes4[1,1].legend(fontsize=8);  axes4[1,1].grid(True, alpha=0.25)

for ax_t in axes4.flat:
    ax_t.set_xlabel('Time (second)', fontsize=8)
    for t0, t1 in exp_windows:
        ax_t.axvspan(t0, t1, alpha=0.1, color='purple')

fig4.tight_layout()
fig4.savefig('case1_velocity_lag.png', dpi=150, bbox_inches='tight')
print('  Saved: case1_velocity_lag.png')

# ── Figure 5: Zoom on first exposure window ────────────────────────────────
tw0, tw1 = exp_windows[0]
pad = 10e-3
zm  = (t_s >= tw0-pad) & (t_s <= tw1+pad)
t_zm = t_s_sec[zm]

fig5, axes5 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig5.suptitle('Case 1 — First Exposure Window (Zoomed)\n'
              'Spring-coupling lag causes non-zero e_syn despite PID+FF',
              fontsize=11, fontweight='bold')

axes5[0].plot(t_zm, e_syn[zm], '#d62728', lw=1.2, label=r'$e_{syn}$')
axes5[0].plot(t_zm, ma_syn[zm], '#1f77b4', lw=1.5, ls='--', label=r'$MA_{syn}$')
axes5[0].axhline( 1.0,  color='k', ls=':', lw=1.0)
axes5[0].axhline(-1.25, color='k', ls=':', lw=1.0)
axes5[0].axvspan(tw0, tw1, alpha=0.18, color='purple', label='Exposure')
axes5[0].set_ylabel('Error (nm)');  axes5[0].legend(fontsize=8);  axes5[0].grid(True, alpha=0.3)

axes5[1].plot(t_zm, msd_syn[zm], '#ff7f0e', lw=1.2, label=r'$MSD_{syn}$')
axes5[1].axhline(7.0, color='r', ls='--', lw=1.0, label='7 nm spec')
axes5[1].axvspan(tw0, tw1, alpha=0.18, color='purple')
axes5[1].set_ylabel('MSD (nm)');  axes5[1].legend(fontsize=8);  axes5[1].grid(True, alpha=0.3)

axes5[2].plot(t_zm, lag_w[zm], '#1f77b4', lw=1.2, label='Wafer lag (nm)')
axes5[2].plot(t_zm, lag_r[zm], '#ff7f0e', lw=1.2, label='Reticle lag (nm)')
axes5[2].axvspan(tw0, tw1, alpha=0.18, color='purple')
axes5[2].set_ylabel('Spring lag (nm)');  axes5[2].set_xlabel('Time (second)')
axes5[2].legend(fontsize=8);  axes5[2].grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig('case1_exposure_zoom.png', dpi=150, bbox_inches='tight')
print('  Saved: case1_exposure_zoom.png')

# ── Summary ────────────────────────────────────────────────────────────────
print('\n' + '═'*64)
print('Case 1 Simulation Summary')
print('═'*64)
print(f'  Fine stages      : NONE (Case 1)')
print(f'  Control          : collocated PID+inertia FF, ωc = {OMC/(2*np.pi):.0f} Hz')
print(f'  Dies             : {N_DIES} × {int(DIE_Y*1e3)}×{int(DIE_X*1e3)} mm²  (1 row × 4 columns)')
print(f'  Wafer scan       : {V_W_SCAN*1e3:.0f} mm/s,  a_max = {A_W_MAX:.0f} m/s²')
print(f'  Reticle scan     : {V_R_SCAN*1e3:.0f} mm/s,  a_max = {A_R_MAX:.0f} m/s²  (opposite)')
print(f'  T_sim            : {T_SIM:.3f} s  ({T_SIM*1e3:.0f} ms)')
print()
print('  Key dynamic parameters:')
print(f'    Wafer   2-mass mode: {ω_WE/(2*np.pi):.1f} Hz, ζ={ζ_WE:.3f}')
print(f'    Reticle 2-mass mode: {ω_RE/(2*np.pi):.1f} Hz, ζ={ζ_RE:.4f} ← lightly damped')
print(f'    Spring lag (wafer,  a=20 m/s²):  {M_WE*A_W_MAX/K_W*1e9:.1f} nm')
print(f'    Spring lag (reticle,a=80 m/s²):  {M_RE*A_R_MAX/K_R*1e9:.1f} nm')
print()
print('  Performance during exposure:')
print(f'    MA_syn  ∈ [{ma_exp.min():+.2f}, {ma_exp.max():+.2f}] nm  '
      f'(spec: [−1.25, +2] nm)')
print(f'    MSD_syn max = {msd_exp.max():.2f} nm  (spec: ≤ 7 nm)')
print('═'*64)
