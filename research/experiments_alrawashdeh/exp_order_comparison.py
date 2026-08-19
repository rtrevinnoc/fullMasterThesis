"""Profile order x controller comparison.

Does trajectory smoothness (bounded acceleration -> jerk -> snap) improve
tracking, and does it still matter once the controller improves? Runs the
exact generators of profiles.py at the same kinematic bounds under two
controllers: Case 1 (bare collocated PID, the baseline) and Case 3b
(feedforward + fine-stage HIGS).

Usage: python3 exp_order_comparison.py <fig_dir>
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rawashdeh_control as lc  # noqa: E402
import profiles  # noqa: E402

fig_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(fig_dir, exist_ok=True)

V, A, J, S = 0.8, 20.0, 1600.0, 1e5
S_TIGHT = 2e4
DIE_L = 0.032
DIES = [(-0.048, 0.0), (-0.016, 0.0), (0.016, 0.0), (0.048, 0.0)]
T_EXP = 5.5e-3 / (V / lc.ALPHA)
WINDOW = max(1, int(T_EXP / lc.DT))

PROFILES = [
    ("2nd order (accel-bounded)", profiles.factory(2)),
    ("3rd order (jerk-bounded)", profiles.factory(3)),
    ("4th order (snap-bounded)", profiles.factory(4)),
    (f"4th order, tight snap ({S_TIGHT:g})", profiles.factory(4, s_tight=S_TIGHT)),
]
CONTROLLERS = [("case1", "Case 1 (PID only)"), ("case3b", "Case 3b (FF + HIGS)")]

print("Calibrating feedforward ...", flush=True)
ff_w, ff_r = lc.calibrate_lag(V, A, J, S)

results = {}
traces = {}
for pname, pf in PROFILES:
    t_scan = pf(V, A, J, S, 0.074).t_total
    for cfg, cname in CONTROLLERS:
        per_die = lc.run_wafer(DIES, DIE_L, V, A, J, S, cfg,
                               ff_w=ff_w, ff_r=ff_r, collect="all",
                               profile_factory=pf)
        stats = []
        for seg in per_die:
            cruise = seg[len(seg) // 3: 2 * len(seg) // 3]
            stats.append(lc.moving_stats(cruise, WINDOW))
        ma = max(x[0] for x in stats)
        msd = max(x[1] for x in stats)
        full_msd = max(lc.moving_stats(seg, WINDOW)[1] for seg in per_die)
        results[(pname, cfg)] = (ma, msd, full_msd, t_scan)
        traces[(pname, cfg)] = per_die[1]
        print(f"[{pname:<28s} | {cfg}] t_scan={t_scan*1e3:6.1f}ms  "
              f"cruise MA={ma*1e9:>12,.1f}nm  cruise MSD={msd*1e9:>11,.1f}nm  "
              f"full-scan MSD={full_msd*1e9:>12,.1f}nm", flush=True)

print("\nSUMMARY")
print(f"{'profile':<30s} {'ctrl':<8s} {'t_scan':>8s} {'cruise MA':>13s} "
      f"{'cruise MSD':>13s} {'full MSD':>13s}")
for (pname, cfg), (ma, msd, fmsd, ts) in results.items():
    print(f"{pname:<30s} {cfg:<8s} {ts*1e3:>6.1f}ms {ma*1e9:>11,.1f}nm "
          f"{msd*1e9:>11,.1f}nm {fmsd*1e9:>11,.1f}nm")

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
colors = {"2nd": "#c0392b", "3rd": "#e67e22", "4th order (snap-bounded)": "#0A3D62",
          "4th order, tight": "#27ae60"}
for ax, (cfg, cname) in zip(axes, CONTROLLERS):
    for pname, _ in PROFILES:
        key = next(k for k in colors if pname.startswith(k))
        seg = traces[(pname, cfg)]
        t_ms = np.arange(len(seg)) * lc.DT * 1e3
        ax.plot(t_ms, np.array(seg) * 1e9, lw=0.8, label=pname,
                color=colors[key])
    ax.set_title(cname, fontsize=9, loc="left")
    ax.set_ylabel("$e_{syn}$ [nm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right")
axes[-1].set_xlabel("time within scan [ms]", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "order_comparison_esyn.png"), dpi=160,
            bbox_inches="tight")
print(f"\nSaved {os.path.join(fig_dir, 'order_comparison_esyn.png')}")
