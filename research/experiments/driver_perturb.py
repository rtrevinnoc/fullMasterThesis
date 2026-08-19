"""Perturbation-injection experiment, cruise-segment criteria.

Per die, e_syn is recorded during the middle third of the scan (the
constant-velocity cruise that contains the exposure). Criteria, both
offset-immune:
  MA-shift : |mean(e_syn) - baseline mean of the same die| > 100 nm
  MSD      : max moving-std > 2x the baseline wafer median
Disturbances are forces on the wafer end-effector.

Usage: python3 driver_perturb_v3.py <out_dir>
"""
import sys
import types
import os

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

sys.modules["glfw"] = types.ModuleType("glfw")
import mujoco  # noqa: E402

viewer_stub = types.ModuleType("mujoco.viewer")
sys.modules["mujoco.viewer"] = viewer_stub
mujoco.viewer = viewer_stub

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

CNN_DIR = "/Users/rtrevinnoc/maestria/tesis/research/nn/cnn"
sys.path.insert(0, CNN_DIR)
os.chdir(CNN_DIR)
sys.path.append(os.path.abspath(os.path.join(CNN_DIR, "../../simulation")))
import litho_sim  # noqa: E402
import test_pipeline  # noqa: E402

MA_MARGIN = 100e-9
V_SCAN, A_MAX, J_MAX, S_MAX = 0.8, 20.0, 1600.0, 1e5
DIE_AREA = 256.0
SCRATCH_X = 0.064
VIB_FORCE_SIGMA = 6.0
DRIFT_FORCE = 30.0
SCRATCH_FORCE = 60.0

die_l = np.sqrt(DIE_AREA) * 1e-3
wafer_r = 0.150
alpha = 0.25
dies = litho_sim.get_wafer_dies(wafer_r, die_l)
rng = np.random.default_rng(42)


def moving_std(x, w):
    if len(x) < w:
        return np.array([np.std(x)])
    ma = np.convolve(x, np.ones(w) / w, mode="valid")
    ma2 = np.convolve(np.asarray(x) ** 2, np.ones(w) / w, mode="valid")
    return np.sqrt(np.maximum(ma2 - ma ** 2, 0.0))


def run_sim(perturbation):
    controller = litho_sim.StepperController(dies, die_l, V_SCAN, A_MAX,
                                             J_MAX, S_MAX, alpha)
    die_grid_xml = ""
    for cx, cy in dies:
        die_grid_xml += (
            f'<geom type="box" size="{die_l/2 - 0.0005} {die_l/2 - 0.0005} 0.0011" '
            f'pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'
        )
    xml = litho_sim.get_model_xml(die_l, die_grid_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    cancel_reactions = litho_sim.make_reaction_canceller(model)
    wafer_body = model.body("wafer").id

    t_exp = 5.5e-3 / (V_SCAN / alpha)
    window = max(1, int(t_exp / model.opt.timestep))
    max_sim_time = len(dies) * (controller.profile.t_total + controller.t_step + 0.1)

    per_die = []          # (mean, max moving-std) on cruise segment
    cur = []
    last_die_idx = -1
    prev_ref = None
    while data.time < max_sim_time:
        xw, yw, xr, yr = controller.get_ref(data.time)
        data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
        data.ctrl[6] = xr; data.ctrl[7] = yr; data.ctrl[8] = 0
        if prev_ref is not None:      # LS error-velocity damping references
            dt = model.opt.timestep
            data.ctrl[12] = (xw - prev_ref[0]) / dt
            data.ctrl[13] = (yw - prev_ref[1]) / dt
            data.ctrl[14] = (xr - prev_ref[2]) / dt
            data.ctrl[15] = (yr - prev_ref[3]) / dt
        prev_ref = (xw, yw, xr, yr)
        cancel_reactions(data)
        data.xfrc_applied[wafer_body][:] = 0
        if perturbation == "vibration":
            data.xfrc_applied[wafer_body][0] = rng.normal(0, VIB_FORCE_SIGMA)
            data.xfrc_applied[wafer_body][1] = rng.normal(0, VIB_FORCE_SIGMA)
        elif perturbation == "drift":
            if np.sqrt(xw ** 2 + yw ** 2) > 0.12:
                data.xfrc_applied[wafer_body][1] = DRIFT_FORCE
        elif perturbation == "scratch":
            if abs(xw - SCRATCH_X) < 0.005:
                data.xfrc_applied[wafer_body][1] = SCRATCH_FORCE
        # pre-step pairing: body state and reference both at time t
        if controller.state == "SCANNING":
            e_syn = ((data.body("wafer").xpos[1] - yw)
                     - alpha * (data.body("mask").xpos[1] - yr))
            cur.append(e_syn)
        mujoco.mj_step(model, data)
        if controller.die_idx != last_die_idx and last_die_idx != -1:
            if cur:
                seg = np.array(cur[len(cur) // 3: 2 * len(cur) // 3])
                per_die.append((seg.mean(), moving_std(seg, window).max()))
            else:
                per_die.append((0.0, 0.0))
            cur = []
            if len(per_die) >= len(dies):
                break
        last_die_idx = controller.die_idx
    means = np.array([m for m, _ in per_die])
    msds = np.array([s for _, s in per_die])
    return means, msds


def to_map(status):
    n = int(np.ceil(2 * wafer_r / die_l)) + 2
    wm = np.zeros((2 * n, 2 * n), dtype=int)
    for i in range(-n, n):
        for jj in range(-n, n):
            cx, cy = i * die_l, jj * die_l
            corners = [(cx - die_l / 2, cy - die_l / 2), (cx + die_l / 2, cy - die_l / 2),
                       (cx - die_l / 2, cy + die_l / 2), (cx + die_l / 2, cy + die_l / 2)]
            if all(np.sqrt(x ** 2 + y ** 2) < wafer_r for x, y in corners):
                try:
                    idx = dies.index((cx, cy))
                    wm[jj + n, i + n] = status[idx] if idx < len(status) else 1
                except ValueError:
                    pass
    return wm


cmap = ListedColormap(["#f0f0f0", "#4caf50", "#f44336"])

print(f"Baseline run ({len(dies)} dies)...", flush=True)
base_means, base_msds = run_sim(None)
msd_thresh = 2.0 * np.median(base_msds)
print(f"baseline cruise MSD [nm]: median={np.median(base_msds)*1e9:.1f} "
      f"max={base_msds.max()*1e9:.1f} -> MSD threshold {msd_thresh*1e9:.1f}", flush=True)
print(f"baseline cruise mean spread [nm]: {(base_means.max()-base_means.min())*1e9:.1f}",
      flush=True)

results = []
for name, pert in [("none", None), ("vibration", "vibration"),
                   ("drift", "drift"), ("scratch", "scratch")]:
    means, msds = (base_means, base_msds) if pert is None else run_sim(pert)
    ma_shift = np.abs(means - base_means)
    status = np.where((ma_shift > MA_MARGIN) | (msds > msd_thresh), 2, 1)
    wm = to_map(status)
    pred, conf = test_pipeline.test_inference(wm)
    nfail = int((wm == 2).sum())
    print(f"[{name}] MA-shift[nm]: median={np.median(ma_shift)*1e9:.1f} "
          f"max={ma_shift.max()*1e9:.1f} | MSD[nm]: median={np.median(msds)*1e9:.1f} "
          f"| failing={nfail}/{len(dies)} -> {pred} ({conf:.2%})", flush=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(wm, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(out_dir, f"perturb_map_{name}.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out_dir, f"perturb_{name}.csv"), "w") as f:
        f.write("scan_idx,x,y,ma_shift_nm,msd_nm,fail\n")
        for k, (di, ma, m, fl) in enumerate(zip(dies, ma_shift, msds, status == 2)):
            f.write(f"{k},{di[0]:.3f},{di[1]:.3f},{ma*1e9:.1f},{m*1e9:.1f},{int(fl)}\n")
    results.append((name, pred, conf, nfail))

print("\nSUMMARY (cruise segment, MA-shift > 100 nm or MSD > 2x baseline median)")
for name, pred, conf, nfail in results:
    print(f"{name:>10s} -> {pred:>10s} ({conf:.2%}), failing dies {nfail}/{len(dies)}")
