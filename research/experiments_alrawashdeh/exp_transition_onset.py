"""Transition-region experiment: do trajectory-induced failures concentrate
at row-change dies?

Every die scans the same profile, so what differentiates dies spatially is
the step that precedes their scan: dies scanned right after a row change
(long diagonal repositioning) enter the scan with a larger residual
transient than mid-row dies (short uniform steps). As the settling budget
t_step shrinks (throughput rises), those dies should fail first, producing
an edge-concentrated, scanner-plausible pattern that emerges from the
trajectory itself - no injected disturbances.

Per die: max moving-window MSD of e_syn over the FULL scan (exposure with
zero settle margin). Fixed threshold = 2x the median of the most-settled
run. Reports failing count, row-change enrichment, CNN class, corrected
cost, and wafer cycle time.

Usage: python3 exp_transition.py <fig_dir> [config]
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
from matplotlib.colors import ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rawashdeh_control as lc  # noqa: E402
import costfn  # noqa: E402
import rawashdeh_ss_sim  # noqa: E402

fig_dir = sys.argv[1] if len(sys.argv) > 1 else "."
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "case3b"
os.makedirs(fig_dir, exist_ok=True)

# RELATIVE-ONSET VARIANT (kept for reference): baseline snap + a threshold
# referenced to the settled run, so the row-change spatial onset is visible.
# The main exp_transition.py uses the absolute 7 nm spec, under which the
# controller absorbs the settling (no onset) -- that is the reported result.
V, A, J, S = 0.8, 20.0, 1600.0, 1e5    # baseline snap
DIE_L = 0.016
WAFER_R = 0.150
T_STEPS = [0.20, 0.19, 0.18, 0.175, 0.17, 0.165, 0.16, 0.15, 0.12]
T_EXP = 5.5e-3 / (V / lc.ALPHA)
WINDOW = max(1, int(T_EXP / lc.DT))

dies = rawashdeh_ss_sim.get_wafer_dies(WAFER_R, DIE_L)
post_row_change = np.zeros(len(dies), dtype=bool)
for k in range(1, len(dies)):
    post_row_change[k] = dies[k][1] != dies[k - 1][1]
frac_prc = post_row_change.mean()
print(f"{len(dies)} dies, {int(post_row_change.sum())} post-row-change "
      f"({frac_prc:.1%}), config={CONFIG}", flush=True)

print("Calibrating feedforward ...", flush=True)
ff_w, ff_r = lc.calibrate_lag(V, A, J, S)
print(f"  ff_w={tuple(round(x, 6) for x in ff_w)} "
      f"ff_r={tuple(round(x, 6) for x in ff_r)}", flush=True)

cmap = ListedColormap(["#f0f0f0", "#4caf50", "#f44336"])
threshold = None      # relative onset: 2x the median MSD of the most-settled run
rows = []
for t_step in T_STEPS:
    per_die = lc.run_wafer(dies, DIE_L, V, A, J, S, CONFIG,
                           ff_w=ff_w, ff_r=ff_r, collect="all", t_step=t_step)
    msd = np.array([lc.moving_stats(seg, WINDOW)[1] for seg in per_die])
    if threshold is None:                       # most-settled run defines it
        threshold = 2.0 * np.median(msd)
        print(f"threshold = 2 x median MSD @ t_step={T_STEPS[0]:.2f} "
              f"= {threshold*1e9:,.1f} nm", flush=True)
    status = np.where(msd > threshold, 2, 1)
    wm = costfn.dies_to_map(dies, DIE_L, WAFER_R, status)
    j_map, info = costfn.map_cost(wm)
    fail = status == 2
    nfail = int(fail.sum())
    enrich = (fail[post_row_change].mean() / max(fail.mean(), 1e-12)
              if nfail else 0.0)
    # scan-order correlation: mean scan index of failing dies, normalized by
    # the mean scan index of all dies (1.0 = uniform; <1 = early dies fail)
    idx = np.arange(len(dies))
    order = (idx[fail].mean() / idx.mean()) if nfail else float("nan")
    with open(os.path.join(fig_dir, f"transition_{int(t_step*1e3)}ms.csv"),
              "w") as f:
        f.write("scan_idx,x,y,msd_nm,fail\n")
        for k, (d, m, fl) in enumerate(zip(dies, msd, fail)):
            f.write(f"{k},{d[0]:.3f},{d[1]:.3f},{m*1e9:.1f},{int(fl)}\n")
    t_scan_total = lc.rawashdeh_ss_sim.StepperController(
        dies, DIE_L, V, A, J, S, lc.ALPHA).profile.t_total
    t_wafer = len(dies) * (t_scan_total + t_step)
    rows.append((t_step, nfail, enrich, order, info["label"], info["conf"],
                 j_map, t_wafer))
    print(f"[t_step={t_step*1e3:.0f}ms] MSD median={np.median(msd)*1e9:,.1f}nm "
          f"| failing={nfail}/{len(dies)} "
          f"| row-change enrich={enrich:.2f} | scan-order={order:.2f} "
          f"| CNN={info['label']} ({info['conf']:.2%}) "
          f"| J_map={j_map:.4f} | T_wafer={t_wafer:.1f}s", flush=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(wm, cmap=cmap, vmin=0, vmax=2, origin="lower",
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(fig_dir, f"transition_map_{int(t_step*1e3)}ms.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

print("\nSUMMARY (fixed threshold, decreasing settle budget)")
print(f"{'t_step':>8s} {'failing':>9s} {'enrich':>7s} {'order':>6s} "
      f"{'CNN':>10s} {'J_map':>7s} {'T_wafer':>8s}")
for t_step, nfail, enrich, order, label, conf, j_map, t_wafer in rows:
    print(f"{t_step*1e3:>6.0f}ms {nfail:>5d}/{len(dies)} {enrich:>7.2f} "
          f"{order:>6.2f} {label:>10s} {j_map:>7.4f} {t_wafer:>7.1f}s")
