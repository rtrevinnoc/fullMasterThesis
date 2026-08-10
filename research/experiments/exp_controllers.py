"""Controller-ladder experiment: Case 1 -> Case 2 (FF) -> Case 3a (SS PI)
-> Case 3b (SS HIGS) on the baseline trajectory, 4-die row.

Reports per-config cruise-window max|MA| / max MSD of the offset-corrected
synchronization error, and saves a comparison figure.

Usage: python3 exp_controllers.py <fig_dir>
"""
import sys
import types
import os

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
import litho_control as lc  # noqa: E402

fig_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(fig_dir, exist_ok=True)

V, A, J, S = 0.8, 20.0, 1600.0, 1e5
DIE_L = 0.032
DIES = [(-0.048, 0.0), (-0.016, 0.0), (0.016, 0.0), (0.048, 0.0)]
T_EXP = 5.5e-3 / (V / lc.ALPHA)
WINDOW = max(1, int(T_EXP / lc.DT))

print("Calibrating lag model lag = lv*v + la*a ...", flush=True)
ff_w, ff_r = lc.calibrate_lag(V, A, J, S, verbose=True)
print(f"  wafer  : lv = {ff_w[0]*1e3:.4f} mm/(m/s)   la = {ff_w[1]*1e6:.3f} um/(m/s^2)",
      flush=True)
print(f"  reticle: lv = {ff_r[0]*1e3:.4f} mm/(m/s)   la = {ff_r[1]*1e6:.3f} um/(m/s^2)",
      flush=True)

CONFIGS = ["case1", "case2", "case3a", "case3b"]
LABELS = {
    "case1": "Case 1: LS PID only",
    "case2": "Case 2: + spring-compensation FF",
    "case3a": "Case 3a: + SS-stage PI",
    "case3b": "Case 3b: + SS-stage HIGS",
}

results = {}
traces = {}
for cfg in CONFIGS:
    per_die = lc.run_wafer(DIES, DIE_L, V, A, J, S, cfg,
                           ff_w=ff_w, ff_r=ff_r, collect="all")
    stats = []
    for seg in per_die:
        cruise = seg[len(seg) // 3: 2 * len(seg) // 3]
        stats.append(lc.moving_stats(cruise, WINDOW))
    ma = max(s[0] for s in stats)
    msd = max(s[1] for s in stats)
    results[cfg] = (ma, msd)
    traces[cfg] = per_die
    print(f"[{cfg}] cruise max|MA| = {ma*1e9:,.2f} nm   "
          f"cruise max MSD = {msd*1e9:,.2f} nm", flush=True)

print("\nSUMMARY (spec: |MA| within [-1.25,+2] nm, MSD <= 7 nm)")
for cfg in CONFIGS:
    ma, msd = results[cfg]
    ok = "PASS" if (ma < 2e-9 and msd < 7e-9) else "fail"
    print(f"  {LABELS[cfg]:<38s} MA {ma*1e9:>12,.2f} nm  "
          f"MSD {msd*1e9:>12,.2f} nm  [{ok}]")

fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=False)
for ax, cfg in zip(axes, CONFIGS):
    seg = traces[cfg][1]           # second die: steady-state behavior
    t_ms = np.arange(len(seg)) * lc.DT * 1e3
    ax.plot(t_ms, np.array(seg) * 1e9, lw=0.7, color="#0A3D62")
    n = len(seg)
    ax.axvspan(t_ms[n // 3], t_ms[2 * n // 3], color="gold", alpha=0.25,
               label="cruise (exposure)")
    ax.set_title(LABELS[cfg], fontsize=9, loc="left")
    ax.set_ylabel("$e_{syn}$ [nm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("time within scan [ms]", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "controllers_esyn.png"), dpi=160,
            bbox_inches="tight")
print(f"\nSaved {os.path.join(fig_dir, 'controllers_esyn.png')}")
