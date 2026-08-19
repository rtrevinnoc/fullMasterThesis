"""Experiment 1: reproduction of the paper's actual Case 1 study.

Al-Rawashdeh et al. (2022), Sec. 6.1 "Case 1": the full multi-body machine
(long-stroke -> short-stroke -> stage -> end-effector per chain) with ZERO
fine stages, controlled by a collocated PID at the long-stroke encoder only.
This is NOT a reduced two-body-per-chain model -- it is realized directly on
the real full MuJoCo model (research/simulation/rawashdeh_ss_sim.py) via
litho_control's existing "case1" configuration (LS position tracking only,
no feedforward, no fine-stage loop). The headline MA/MSD numbers are pulled
from the same code path exp_controllers.py uses, so this experiment reports
exactly the same Case-1 numbers as the rest of the controller ladder; the
extra instrumentation here only adds the continuous-trace plots (references,
tracking error, spring lag, MA/MSD time series) that the ladder script
doesn't produce.

Usage: python3 exp_case1_full.py <fig_dir>
"""
import sys
import types
import os
import math

sys.modules["glfw"] = types.ModuleType("glfw")
import mujoco  # noqa: E402

viewer_stub = types.ModuleType("mujoco.viewer")
viewer_stub.launch = viewer_stub.launch_passive = lambda *a, **k: None
sys.modules["mujoco.viewer"] = viewer_stub
mujoco.viewer = viewer_stub

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawashdeh_control as lc  # noqa: E402
import rawashdeh_ss_sim as litho_sim  # noqa: E402

fig_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(fig_dir, exist_ok=True)

V, A, J, S = 0.8, 20.0, 1600.0, 1e5
DIE_L = 0.032
DIES = [(-0.048, 0.0), (-0.016, 0.0), (0.016, 0.0), (0.048, 0.0)]
ALPHA = lc.ALPHA
DT = lc.DT
T_EXP = 5.5e-3 / (V / ALPHA)
WINDOW = max(1, int(T_EXP / DT))
MA_UPPER, MA_LOWER, MSD_UPPER = 2.0e-9, -1.25e-9, 7.0e-9

# ---------------------------------------------------------------------------
# Resonance report -- meaningful post stage-mass fix (was ~825 Hz pre-fix)
# ---------------------------------------------------------------------------
f_stage = math.sqrt(litho_sim.K_ST_F / litho_sim.M_ST_F) / (2 * math.pi)
print(f"SS->Stage corner frequency (both chains, K_ST_F/M_ST_F): "
      f"{f_stage:.1f} Hz")

# ---------------------------------------------------------------------------
# Headline MA/MSD: same code path as exp_controllers.py's case1 row, so this
# experiment's numbers are guaranteed identical to the rest of the ladder.
# ---------------------------------------------------------------------------
per_die_headline = lc.run_wafer(DIES, DIE_L, V, A, J, S, "case1",
                                collect="all")
stats = [lc.moving_stats(np.asarray(seg)[len(seg) // 3: 2 * len(seg) // 3],
                         WINDOW) for seg in per_die_headline]
ma_headline = max(s[0] for s in stats)
msd_headline = max(s[1] for s in stats)
print(f"[case1] cruise max|MA| = {ma_headline*1e9:,.2f} nm   "
      f"cruise max MSD = {msd_headline*1e9:,.2f} nm")

# ---------------------------------------------------------------------------
# Continuous full-trace run (hold + all dies' scan+step), config=case1.
# rawashdeh_control._run only returns per-die SCANNING segments and discards
# STEPPING phases and absolute traces, so this loop is a local instrumented
# copy of that same control law -- same plant, same config, richer logging.
# ---------------------------------------------------------------------------
controller = lc.StepperControllerSS(DIES, DIE_L, V, A, J, S, ALPHA)
plant = litho_sim.StateSpaceLithoPlant()

x = np.zeros(96)
u = np.zeros(6)

hold_wy, hold_ry, hold_wx = [], [], []
t = 0.0
controller.state = "HOLD"
while t < lc.HOLD_TIME:
    xw0, yw0, xr0, yr0 = controller.get_ref(t)
    u[0] = 1e7 * (yw0 - x[plant.IDX_WAFR_Y])
    u[3] = 1e7 * (yr0 - x[plant.IDX_MASK_Y])
    x = plant.step(x, u, lc.DT)
    t += lc.DT
    if t > lc.HOLD_TIME - 0.05:
        hold_wy.append(x[plant.IDX_WAFR_Y] - yw0)
        hold_ry.append(x[plant.IDX_MASK_Y] - yr0)
        hold_wx.append(x[3*15] - xw0)

off_wy = float(np.mean(hold_wy))
off_ry = float(np.mean(hold_ry))
off_wx = float(np.mean(hold_wx))

t_origin = t
max_time = t_origin + len(DIES) * (controller.profile.t_total + controller.t_step + 0.1)

t_l, ywref_l, xwref_l, yrref_l = [], [], [], []
ew_l, er_l, ewx_l = [], [], []
wstage_l, rstage_l, wafer_l, mask_l = [], [], [], []
scanning_l = []

controller.t_start = t
controller.state = "SCANNING"
controller.die_idx = 0

while t < max_time and controller.state != "DONE":
    t_rel = t - t_origin
    xw, yw, xr, yr, vw, aw, vr, ar, vxw = lc.ref_kin(controller, t)

    u[0] = 1e7 * (yw - x[plant.IDX_WAFR_Y])
    u[3] = 1e7 * (yr - x[plant.IDX_MASK_Y])

    wafer_y = x[plant.IDX_WAFR_Y]
    mask_y = x[plant.IDX_MASK_Y]
    w_st = x[3*14 + 1]
    r_st = x[3*5 + 1]

    e_w = wafer_y - off_wy - yw
    e_r = mask_y - off_ry - yr
    e_wx = x[3*15] - off_wx - xw

    t_l.append(t_rel)
    ywref_l.append(yw); xwref_l.append(xw); yrref_l.append(yr)
    ew_l.append(e_w); er_l.append(e_r); ewx_l.append(e_wx)
    wstage_l.append(w_st)
    rstage_l.append(r_st)
    wafer_l.append(wafer_y); mask_l.append(mask_y)
    scanning_l.append(controller.state == "SCANNING")

    x = plant.step(x, u, lc.DT)
    t += lc.DT

t_arr = np.array(t_l)
yw_ref = np.array(ywref_l); xw_ref = np.array(xwref_l); yr_ref = np.array(yrref_l)
e_w = np.array(ew_l); e_r = np.array(er_l); e_wx = np.array(ewx_l)
w_stage = np.array(wstage_l); r_stage = np.array(rstage_l)
wafer_y = np.array(wafer_l); mask_y = np.array(mask_l)
scanning = np.array(scanning_l)
e_syn = e_w - ALPHA * e_r

# Exposure windows: middle third of each contiguous SCANNING run, matching
# the "cruise" convention used everywhere else in this codebase
# (exp_controllers.py, tune_fine.py).
exp_windows = []
edges = np.flatnonzero(np.diff(scanning.astype(int)))
starts = [0] if scanning[0] else []
starts += [i + 1 for i in edges if scanning[i + 1]]
ends = [i + 1 for i in edges if not scanning[i + 1]]
if scanning[-1]:
    ends.append(len(scanning))
for s0, s1 in zip(starts, ends):
    n = s1 - s0
    a0, a1 = s0 + n // 3, s0 + 2 * n // 3
    exp_windows.append((t_arr[a0], t_arr[a1]))

# ---------------------------------------------------------------------------
# MA/MSD time series (uniform dt=DT, so a plain sliding window suffices --
# same cumulative-sum construction as the retired reduced-model script).
# ---------------------------------------------------------------------------
def moving_ma_msd(sig, window):
    hw = max(1, int(round(window / 2)))
    pad = np.pad(sig, hw, mode="edge")
    cs = np.cumsum(pad)
    cs2 = np.cumsum(pad ** 2)
    n = 2 * hw + 1
    ma = (cs[2 * hw:] - cs[:len(sig)]) / n
    m2 = (cs2[2 * hw:] - cs2[:len(sig)]) / n
    msd = np.sqrt(np.maximum(m2 - ma ** 2, 0.0))
    return ma, msd


ma_syn, msd_syn = moving_ma_msd(e_syn, WINDOW)
in_exp = np.zeros(len(t_arr), dtype=bool)
for t0, t1 in exp_windows:
    in_exp |= (t_arr >= t0) & (t_arr <= t1)
ma_exp, msd_exp = ma_syn[in_exp], msd_syn[in_exp]
print(f"Continuous trace: MA_syn during exposure in "
      f"[{ma_exp.min()*1e9:+.2f}, {ma_exp.max()*1e9:+.2f}] nm  "
      f"(spec [-1.25, +2] nm)")
print(f"Continuous trace: MSD_syn during exposure max = "
      f"{msd_exp.max()*1e9:.2f} nm  (spec <= 7 nm)")

# ---------------------------------------------------------------------------
# Figures (same 5 filenames as the retired reduced-model script, so
# experimentsChapter.tex's \\includegraphics paths need no changes)
# ---------------------------------------------------------------------------

# Fig 12 -- desired step-and-scan reference trajectories
fig1, (ax12a, ax12b) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax12a.plot(t_arr, yw_ref, color="#1f77b4", lw=1.0, label="Desired scan position")
ax12a.set_ylabel("Y position (m)"); ax12a.legend(fontsize=9, loc="upper right")
ax12a.grid(True, alpha=0.25)
ax12b.plot(t_arr, xw_ref, color="#1f77b4", lw=1.0, label="Desired step position")
ax12b.set_ylabel("X position (m)"); ax12b.set_xlabel("Time (s)")
ax12b.legend(fontsize=9, loc="upper left"); ax12b.grid(True, alpha=0.25)

fig1.tight_layout()
fig1.savefig(os.path.join(fig_dir, "case1_fig12_trajectories.png"), dpi=150,
            bbox_inches="tight")
print("Saved case1_fig12_trajectories.png")

# Tracking errors (diagnostic)
fig2, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax_t, err, lbl, col in [
    (axes[0], e_w * 1e9, "Wafer $e_{w,y}$ (nm)", "#1f77b4"),
    (axes[1], e_r * 1e9, "Reticle $e_{r,y}$ (nm)", "#ff7f0e"),
    (axes[2], e_wx * 1e9, "Wafer $e_{w,x}$ (nm)", "#2ca02c"),
    (axes[3], e_syn * 1e9, "Sync. $e_{syn}$ (nm)", "#d62728"),
]:
    ax_t.plot(t_arr, err, color=col, lw=0.6)
    ax_t.axhline(0, color="k", lw=0.5, ls="--")
    ax_t.set_ylabel(lbl, fontsize=11); ax_t.grid(True, alpha=0.25)
    for t0, t1 in exp_windows:
        ax_t.axvspan(t0, t1, alpha=0.18, color="purple")
axes[3].set_xlabel("Time (s)", fontsize=11)

fig2.savefig(os.path.join(fig_dir, "case1_tracking_errors.png"), dpi=150,
            bbox_inches="tight")
print("Saved case1_tracking_errors.png")

# Fig 14 -- MA/MSD performance indices
fig3, (axma, axmsd) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
MA_LIM = max(1e-6, float(np.max(np.abs(ma_syn))) * 1.1)
MSD_LIM = max(2.5e-8, float(np.max(msd_syn)) * 1.1)
axma.plot(t_arr, ma_syn, color="#1f4e79", lw=0.7, label="MA (Case 1, no fine stage)")
axma.axhline(MA_UPPER, color="#c00000", ls="--", lw=1.0, label="MA upper spec")
axma.axhline(MA_LOWER, color="#843c0c", ls="--", lw=1.0, label="MA lower spec")
axma.set_ylabel("Moving average (m)", fontsize=11)
axma.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axma.set_ylim(-MA_LIM, MA_LIM); axma.legend(fontsize=9, loc="upper right")
axma.grid(True, alpha=0.2)
axmsd.plot(t_arr, msd_syn, color="#1f4e79", lw=0.7, label="MSD (Case 1, no fine stage)")
axmsd.axhline(MSD_UPPER, color="#c00000", ls="--", lw=1.0, label="MSD spec")
axmsd.set_ylabel("Moving std. dev. (m)", fontsize=11); axmsd.set_xlabel("Time (s)")
axmsd.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axmsd.set_ylim(0, MSD_LIM); axmsd.legend(fontsize=9, loc="upper right")
axmsd.grid(True, alpha=0.2)

fig3.tight_layout()
fig3.savefig(os.path.join(fig_dir, "case1_fig14_ma_msd.png"), dpi=150,
            bbox_inches="tight")
print("Saved case1_fig14_ma_msd.png")

# Velocities + coarse(Stage)-vs-end-effector spring lag across the final K8
# coupling -- the closest full-model analog of the retired reduced model's
# coarse-vs-end-effector lag panel.
lag_w = (w_stage - wafer_y) * 1e9
lag_r = (r_stage - mask_y) * 1e9
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
axes4[0].plot(t_arr, np.gradient(wafer_y, DT), "#1f77b4", lw=0.8,
             label="Wafer end-effector")
axes4[0].plot(t_arr, np.gradient(w_stage, DT), "#aec7e8", lw=0.5, alpha=0.8,
             label="Wafer stage (upstream of K8)")
axes4[0].set_ylabel("v (m/s)"); axes4[0].set_title("Wafer scan (y) velocity")
axes4[0].legend(fontsize=8); axes4[0].grid(True, alpha=0.25)
axes4[1].plot(t_arr, lag_w, "#1f77b4", lw=0.9, label="Wafer K8 spring lag (nm)")
axes4[1].plot(t_arr, lag_r, "#ff7f0e", lw=0.9, label="Reticle K8 spring lag (nm)")
axes4[1].set_ylabel("Stage - end-effector (nm)")
axes4[1].set_title("Final spring-coupling lag"); axes4[1].legend(fontsize=8)
axes4[1].grid(True, alpha=0.25)
for ax_t in axes4:
    ax_t.set_xlabel("Time (s)", fontsize=9)
    for t0, t1 in exp_windows:
        ax_t.axvspan(t0, t1, alpha=0.1, color="purple")

fig4.tight_layout()
fig4.savefig(os.path.join(fig_dir, "case1_velocity_lag.png"), dpi=150,
            bbox_inches="tight")
print("Saved case1_velocity_lag.png")

# Exposure-window zoom (first die)
tw0, tw1 = exp_windows[0]
pad = 10e-3
zm = (t_arr >= tw0 - pad) & (t_arr <= tw1 + pad)
fig5, axes5 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes5[0].plot(t_arr[zm], e_syn[zm] * 1e9, "#d62728", lw=1.2, label="$e_{syn}$")
axes5[0].plot(t_arr[zm], ma_syn[zm] * 1e9, "#1f77b4", lw=1.5, ls="--", label="$MA_{syn}$")
axes5[0].axhline(MA_UPPER * 1e9, color="k", ls=":", lw=1.0)
axes5[0].axhline(MA_LOWER * 1e9, color="k", ls=":", lw=1.0)
axes5[0].axvspan(tw0, tw1, alpha=0.18, color="purple", label="Exposure")
axes5[0].set_ylabel("Error (nm)"); axes5[0].legend(fontsize=8); axes5[0].grid(True, alpha=0.3)
axes5[1].plot(t_arr[zm], msd_syn[zm] * 1e9, "#ff7f0e", lw=1.2, label="$MSD_{syn}$")
axes5[1].axhline(MSD_UPPER * 1e9, color="r", ls="--", lw=1.0, label="MSD spec")
axes5[1].axvspan(tw0, tw1, alpha=0.18, color="purple")
axes5[1].set_ylabel("MSD (nm)"); axes5[1].legend(fontsize=8); axes5[1].grid(True, alpha=0.3)
axes5[2].plot(t_arr[zm], lag_w[zm], "#1f77b4", lw=1.2, label="Wafer lag (nm)")
axes5[2].plot(t_arr[zm], lag_r[zm], "#ff7f0e", lw=1.2, label="Reticle lag (nm)")
axes5[2].axvspan(tw0, tw1, alpha=0.18, color="purple")
axes5[2].set_ylabel("Spring lag (nm)"); axes5[2].set_xlabel("Time (s)")
axes5[2].legend(fontsize=8); axes5[2].grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig(os.path.join(fig_dir, "case1_exposure_zoom.png"), dpi=150,
            bbox_inches="tight")
print("Saved case1_exposure_zoom.png")

print("\nSUMMARY")
print(f"  SS->Stage corner frequency: {f_stage:.1f} Hz (paper/diagram target: 180 Hz)")
print(f"  Headline (exp_controllers.py code path) cruise max|MA| = "
      f"{ma_headline*1e9:,.2f} nm, max MSD = {msd_headline*1e9:,.2f} nm")
print(f"  Total sequence duration (hold + {len(DIES)} dies): {t_arr[-1]*1e3:.1f} ms")
print(f"  First exposure window: [{exp_windows[0][0]*1e3:.1f}, "
      f"{exp_windows[0][1]*1e3:.1f}] ms")
