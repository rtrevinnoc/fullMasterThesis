"""Quick gain sweep for the fine-stage loops (PI and HIGS)."""
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
import litho_control as lc  # noqa: E402

V, A, J, S = 0.8, 20.0, 1600.0, 1e5
DIE_L = 0.032
DIES = [(-0.016, 0.0), (0.016, 0.0)]
T_EXP = 5.5e-3 / (V / lc.ALPHA)
WINDOW = max(1, int(T_EXP / lc.DT))

ff_w, ff_r = lc.calibrate_lag(V, A, J, S)
print(f"ff_w={ff_w} ff_r={ff_r}", flush=True)


def score(cfg):
    per_die = lc.run_wafer(DIES, DIE_L, V, A, J, S, cfg,
                           ff_w=ff_w, ff_r=ff_r, collect="all")
    stats = []
    for seg in per_die:
        cruise = seg[len(seg) // 3: 2 * len(seg) // 3]
        stats.append(lc.moving_stats(cruise, WINDOW))
    return max(s[0] for s in stats), max(s[1] for s in stats)


for kp in [20.0, 30.0, 40.0, 60.0, 80.0]:
    for ki in [2e3, 4e3, 8e3, 1.5e4, 2.5e4]:
        lc.FINE_PI = dict(kp=kp, ki=ki)
        ma, msd = score("case3a")
        print(f"PI   kp={kp:<5g} ki={ki:<8g} -> MA {ma*1e9:>12,.1f} nm  "
              f"MSD {msd*1e9:>12,.1f} nm", flush=True)

for kp in [20.0, 30.0, 40.0, 60.0, 80.0]:
    for wi, wh in [(1500.0, 600.0), (2500.0, 1000.0), (4000.0, 1600.0),
                   (2500.0, 400.0), (4000.0, 1000.0)]:
        lc.FINE_HIGS = dict(kp=kp, wi=2 * np.pi * wi, wh=2 * np.pi * wh,
                            kh=1.0)
        ma, msd = score("case3b")
        print(f"HIGS kp={kp:<5g} wi=2pi*{wi:<6g} wh=2pi*{wh:<6g} -> "
              f"MA {ma*1e9:>12,.1f} nm  MSD {msd*1e9:>12,.1f} nm", flush=True)
