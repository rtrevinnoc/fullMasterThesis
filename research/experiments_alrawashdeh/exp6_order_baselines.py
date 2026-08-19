"""Regenerates the 3rd-order/4th-order baseline rows of Table exp6_pso
(experimentsChapter.tex, Experiment 6): the 49-die wafer classification of
the plain baseline trajectory (v=0.8, a=20, j=1600[, s=1e5]) under Case 3b,
same t_step/threshold as exp_pso.py, for comparison against the PSO-searched
E3/E4 optima. No current driver reproduces this specific pair of rows
(exp_pso.py always runs order 4; exp_order_comparison.py never runs the
49-die CNN/J_map classification), so this is a standalone script reusing
the same evaluate()-style logic as exp_pso.py.

Usage: python3 exp6_order_baselines.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawashdeh_control as lc  # noqa: E402
import rawashdeh_ss_sim  # noqa: E402
import costfn  # noqa: E402
import profiles  # noqa: E402

DIE_L = 0.032
WAFER_R = 0.150
T_STEP = 0.15
CONFIG = "case3b"
dies = rawashdeh_ss_sim.get_wafer_dies(WAFER_R, DIE_L)
FF_W, FF_R = lc.calibrate_lag(0.8, 20.0, 1600.0, 1e5)
THRESH = 7e-9


def window_for(v):
    return max(1, int((5.5e-3 / (v / lc.ALPHA)) / lc.DT))


def evaluate(order, v, a, j, s):
    pf = profiles.factory(order)
    per_die = lc.run_wafer(dies, DIE_L, v, a, j, s, CONFIG,
                           ff_w=FF_W, ff_r=FF_R, collect="all", t_step=T_STEP,
                           profile_factory=pf)
    w = window_for(v)
    msd = np.array([lc.moving_stats(seg[len(seg) // 3:2 * len(seg) // 3], w)[1]
                    for seg in per_die])
    status = np.where(msd > THRESH, 2, 1)
    wm = costfn.dies_to_map(dies, DIE_L, WAFER_R, status)
    d_ramp = v * ((v / a) + (a / j))
    t_scan = pf(v, a, j, s, DIE_L + d_ramp).t_total
    t_wafer = len(dies) * (t_scan + T_STEP)
    j_map, info = costfn.map_cost(wm)
    nfail = int((status == 2).sum())
    print(f"order={order} v={v} a={a} j={j} s={s} -> t_scan={t_scan*1e3:.1f}ms "
          f"fail={nfail}/{len(dies)} CNN={info['label']} J_map={j_map:.4f} "
          f"T_wafer={t_wafer:.1f}s", flush=True)


# s is a dummy placeholder for order 3 (StepperController.__init__ needs a
# numeric s_max before profile_factory overrides it; profiles.factory(3)
# ignores it).
evaluate(3, 0.8, 20.0, 1600.0, 1e5)
evaluate(4, 0.8, 20.0, 1600.0, 1e5)
