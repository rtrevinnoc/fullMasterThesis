"""Quick gain sweep for the fine-stage loops (PI and HIGS).

Grid range as of the 2026-08-11 stage-mass fix: the Stage body's mass/inertia
was corrected from the actuator-placeholder values (0.5 kg) to the paper's
m7 (10.5 kg), moving the SS->Stage corner from ~825 Hz to ~180 Hz. That much
lower resonance sits closer to the fine loop's operating range, so the
stability boundary is now much lower gain than the pre-fix grid (kp 24-56)
covered; this grid brackets the new boundary (found empirically at kp~16-18)
instead."""
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


print("--- PI grid ---", flush=True)
for kp in [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]:
    for ki in [kp * 30, kp * 50, kp * 70, kp * 100]:
        lc.FINE_PI = dict(kp=kp, ki=ki)
        ma, msd = score("case3a")
        print(f"PI   kp={kp:<5g} ki={ki:<8g} -> MA {ma*1e9:>12,.1f} nm  "
              f"MSD {msd*1e9:>12,.1f} nm", flush=True)

print("--- HIGS grid ---", flush=True)
for kp in [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]:
    for wi_hz in [80.0, 100.0, 120.0, 150.0, 200.0, 250.0]:
        wh_hz = wi_hz * 0.4
        lc.FINE_HIGS = dict(kp=kp, wi=2 * np.pi * wi_hz, wh=2 * np.pi * wh_hz,
                            kh=1.0)
        ma, msd = score("case3b")
        print(f"HIGS kp={kp:<5g} wi=2pi*{wi_hz:<6g} wh=2pi*{wh_hz:<6g} -> "
              f"MA {ma*1e9:>12,.1f} nm  MSD {msd*1e9:>12,.1f} nm", flush=True)
