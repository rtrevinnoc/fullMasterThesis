"""PSO over the kinematic bounds (v, a, j, s)_max.

Two scenarios sharing plant, controller, threshold, and a throughput reward
(GAMMA * t_wafer / T_REF, so faster is cheaper):
  E3: servo cost      J = mean(per-die max MSD)/threshold + throughput
  E4: yield-aware CNN J = (1 - P_none) + P_ood            + throughput
The servo cost trades throughput against error magnitude; the CNN cost trades
it against pattern acceptability (P_ood is the wall that stops over-aggression),
tolerating scanner-plausible failures a servo cost would spend accuracy to avoid.

The wafer runs with a short settling budget (transition regime) so that the
cost actually varies over the search space. The failure threshold is fixed
from a well-settled reference run and shared by every candidate.

Usage: python3 exp_pso.py <fig_dir> <scenario: e3|e4> [config] [seed]
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
import rawashdeh_ss_sim  # noqa: E402
import costfn  # noqa: E402
import profiles  # noqa: E402

PROFILE_FACTORY = profiles.factory(4)   # exact 15-segment snap-bounded

fig_dir = sys.argv[1]
SCENARIO = sys.argv[2].lower()
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "case3a"
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(fig_dir, exist_ok=True)

DIE_L = 0.032
WAFER_R = 0.150
T_STEP = 0.15            # transition-regime settling budget
T_REF = 15.0             # reference wafer cycle time for the throughput reward
GAMMA = 1.0              # throughput weight: cost carries GAMMA * t_wafer / T_REF
                         # (faster is cheaper; the pattern term is the wall)
BOUNDS = np.array([[0.2, 1.5],        # v [m/s]
                   [5.0, 45.0],       # a [m/s^2]
                   [200.0, 5000.0],   # j [m/s^3]
                   [50.0, 5e5]])      # s [m/s^4]
# s lower bound widened 2026-08-11 (stage-mass fix): the pre-fix bound
# (5e3) sat exactly at the pre-fix spec-compliance boundary, but that
# boundary moved ~100x tighter post-fix (see driver_sweep_spec.py /
# sweep_spec.log: crossing now between s=100 and s=1000, not s~1e3), so the
# old bound no longer reaches any compliant region at all.
N_PARTICLES = 6
N_ITER = 8

dies = rawashdeh_ss_sim.get_wafer_dies(WAFER_R, DIE_L)

print(f"[{SCENARIO}] config={CONFIG} dies={len(dies)} t_step={T_STEP}s "
      f"T_ref={T_REF}s gamma={GAMMA}", flush=True)
print("Calibrating feedforward at baseline ...", flush=True)
FF_W, FF_R = lc.calibrate_lag(0.8, 20.0, 1600.0, 1e5)

def window_for(v):
    return max(1, int((5.5e-3 / (v / lc.ALPHA)) / lc.DT))


THRESH = 7e-9    # absolute spec: exposure-window (cruise) MSD <= 7 nm
print(f"threshold = {THRESH*1e9:.1f} nm (absolute exposure-window spec)", flush=True)

cmap = ListedColormap(["#f0f0f0", "#4caf50", "#f44336"])
history = []


def evaluate(x):
    v, a, j, s = x
    per_die = lc.run_wafer(dies, DIE_L, v, a, j, s, CONFIG,
                           ff_w=FF_W, ff_r=FF_R, collect="all", t_step=T_STEP,
                           profile_factory=PROFILE_FACTORY)
    w = window_for(v)
    # exposure-window (cruise) MSD: middle third of the scan, per the spec
    msd = np.array([lc.moving_stats(seg[len(seg)//3:2*len(seg)//3], w)[1]
                    for seg in per_die])
    status = np.where(msd > THRESH, 2, 1)
    wm = costfn.dies_to_map(dies, DIE_L, WAFER_R, status)
    d_ramp = v * ((v / a) + (a / j))
    t_scan = PROFILE_FACTORY(v, a, j, s, DIE_L + d_ramp).t_total
    t_wafer = len(dies) * (t_scan + T_STEP)
    tput = GAMMA * t_wafer / T_REF        # throughput reward: faster is cheaper
    j_map, info = costfn.map_cost(wm)
    nfail = int((status == 2).sum())
    if SCENARIO == "e4":                   # yield-aware CNN pattern cost
        cost = j_map + tput
    else:                                  # e3: conventional servo cost
        cost = float(np.mean(msd)) / THRESH + tput
    history.append((tuple(x), cost, int((status == 2).sum()), info["label"],
                    j_map, t_wafer))
    print(f"  eval v={v:.3f} a={a:5.1f} j={j:6.0f} s={s:8.0f} "
          f"-> cost={cost:8.4f} fail={int((status == 2).sum()):>2d}/{len(dies)} "
          f"CNN={info['label']:<9s} J_map={j_map:.4f} T={t_wafer:5.1f}s",
          flush=True)
    return cost, wm


import pickle  # noqa: E402

CKPT = os.path.join(fig_dir, f".pso_ckpt_{SCENARIO}_{CONFIG}_{SEED}.pkl")
dim = 4
start_it = 0
if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        ck = pickle.load(f)
    rng, pos, vel, pbest, pbest_cost = (ck["rng"], ck["pos"], ck["vel"],
                                        ck["pbest"], ck["pbest_cost"])
    gbest, gbest_cost, gbest_map = ck["gbest"], ck["gbest_cost"], ck["gbest_map"]
    history = ck["history"]
    start_it = ck["it"] + 1
    print(f"resumed from checkpoint at iteration {start_it}/{N_ITER}", flush=True)
else:
    rng = np.random.default_rng(SEED)
    pos = BOUNDS[:, 0] + rng.random((N_PARTICLES, dim)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    pos[0] = [0.8, 20.0, 1600.0, 1e5]           # seed the baseline
    vel = np.zeros((N_PARTICLES, dim))
    pbest = pos.copy()
    pbest_cost = np.full(N_PARTICLES, np.inf)
    gbest = None
    gbest_cost = np.inf
    gbest_map = None

for it in range(start_it, N_ITER):
    print(f"--- iteration {it+1}/{N_ITER} ---", flush=True)
    for k in range(N_PARTICLES):
        cost, wm = evaluate(pos[k])
        if cost < pbest_cost[k]:
            pbest_cost[k] = cost
            pbest[k] = pos[k].copy()
        if cost < gbest_cost:
            gbest_cost = cost
            gbest = pos[k].copy()
            gbest_map = wm
    for k in range(N_PARTICLES):
        r1, r2 = rng.random(dim), rng.random(dim)
        vel[k] = (0.6 * vel[k]
                  + 1.4 * r1 * (pbest[k] - pos[k])
                  + 1.4 * r2 * (gbest - pos[k]))
        span = BOUNDS[:, 1] - BOUNDS[:, 0]
        vel[k] = np.clip(vel[k], -0.5 * span, 0.5 * span)
        pos[k] = np.clip(pos[k] + vel[k], BOUNDS[:, 0], BOUNDS[:, 1])
    with open(CKPT, "wb") as f:
        pickle.dump(dict(rng=rng, pos=pos, vel=vel, pbest=pbest,
                         pbest_cost=pbest_cost, gbest=gbest,
                         gbest_cost=gbest_cost, gbest_map=gbest_map,
                         history=history, it=it), f)

if os.path.exists(CKPT):
    os.remove(CKPT)

print(f"\nBEST [{SCENARIO}] cost={gbest_cost:.4f} at "
      f"v={gbest[0]:.3f} a={gbest[1]:.2f} j={gbest[2]:.0f} s={gbest[3]:.0f}",
      flush=True)
fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.imshow(gbest_map, cmap=cmap, vmin=0, vmax=2, origin="lower",
          interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
fig.savefig(os.path.join(fig_dir, f"pso_best_map_{SCENARIO}.png"),
            dpi=160, bbox_inches="tight")

with open(os.path.join(fig_dir, f"pso_history_{SCENARIO}.csv"), "w") as f:
    f.write("v,a,j,s,cost,failing,label,j_map,t_wafer\n")
    for (x, cost, nfail, label, j_map, t_wafer) in history:
        f.write(f"{x[0]:.4f},{x[1]:.2f},{x[2]:.0f},{x[3]:.0f},"
                f"{cost:.5f},{nfail},{label},{j_map:.5f},{t_wafer:.2f}\n")
print("history saved", flush=True)
