"""Snap-vs-spec sweep (Exp 3, spec-referenced). Under the full controller
(Case 3b), tighten the snap bound and watch the full 49-die wafer cross the
absolute MSD <= 7 nm exposure-window spec: failing count, CNN class, and the
throughput cost. Fixed v, a, j; the snap bound is the lever.

Usage: python3 driver_sweep_spec.py <out_dir>
"""
import sys
import types
import os

sys.modules["glfw"] = types.ModuleType("glfw")
import mujoco  # noqa: E402
vs = types.ModuleType("mujoco.viewer")
vs.launch = vs.launch_passive = lambda *a, **k: None
sys.modules["mujoco.viewer"] = vs
mujoco.viewer = vs

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import litho_control as lc  # noqa: E402
import litho_sim  # noqa: E402
import costfn  # noqa: E402
import profiles  # noqa: E402

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

V, A, J = 0.8, 20.0, 1600.0
DIE_L, WAFER_R, T_STEP = 0.032, 0.150, 0.20
SPEC = 7e-9
PF = profiles.factory(4)
dies = litho_sim.get_wafer_dies(WAFER_R, DIE_L)
WINDOW = max(1, int((5.5e-3 / (V / lc.ALPHA)) / lc.DT))
cmap = ListedColormap(["#f0f0f0", "#3a7ca5", "#e8c547"])

print(f"{len(dies)} dies, Case 3b, spec = exposure-window MSD <= {SPEC*1e9:.0f} nm", flush=True)
ff_w, ff_r = lc.calibrate_lag(V, A, J, 1e5)

rows = []
for s in [1e5, 2e4, 1e4, 5e3, 1e3, 1e2, 5e1]:
    per = lc.run_wafer(dies, DIE_L, V, A, J, s, "case3b", ff_w=ff_w, ff_r=ff_r,
                       collect="all", t_step=T_STEP, profile_factory=PF)
    cruise = [seg[len(seg)//3:2*len(seg)//3] for seg in per]
    msd = np.array([lc.moving_stats(c, WINDOW)[1] for c in cruise])
    ma = np.array([lc.moving_stats(c, WINDOW)[0] for c in cruise])
    status = np.where(msd > SPEC, 2, 1)
    wm = costfn.dies_to_map(dies, DIE_L, WAFER_R, status)
    _, info = costfn.map_cost(wm)
    t_scan = PF(V, A, J, s, DIE_L + V*((V/A)+(A/J))).t_total
    t_wafer = len(dies) * (t_scan + T_STEP)
    nfail = int((status == 2).sum())
    rows.append((s, t_scan, ma.max(), msd.max(), nfail, info["label"], info["conf"], t_wafer))
    print(f"[s={s:>7.0f}] t_scan={t_scan*1e3:6.1f}ms  cruise max MA={ma.max()*1e9:8.1f}nm  "
          f"max MSD={msd.max()*1e9:8.1f}nm  failing={nfail:>2d}/{len(dies)}  "
          f"CNN={info['label']:<8}({info['conf']:.2f})  T_wafer={t_wafer:5.1f}s", flush=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(wm, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(out_dir, f"specsweep_map_s{int(s)}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

print("\nSUMMARY (snap lever vs the 7 nm exposure-window spec, Case 3b)")
print(f"{'s_max':>8} {'t_scan':>8} {'max MA':>10} {'max MSD':>10} {'fail':>7} {'CNN':>10} {'T_wafer':>8}")
for s, ts, ma, msd, nf, lab, conf, tw in rows:
    print(f"{s:>8.0f} {ts*1e3:>6.1f}ms {ma*1e9:>8.1f}nm {msd*1e9:>8.1f}nm {nf:>4}/{len(dies)} "
          f"{lab:>10} {tw:>7.1f}s")
