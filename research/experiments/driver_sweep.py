"""Sweep the kinematic bounds (v, a, j, s)_max and measure how the simulated
wafer map, the CNN class distribution, and the defect cost respond.

Die criterion: MSD of e_syn during the die's scan, window = exposure time.
MSD is a moving standard deviation, hence immune to the constant geometric
offset present in absolute body positions. Spec: MSD <= 7 nm.

Cost: J = sum of CNN probabilities of the scanner-plausible defect classes
(Scratch, Edge-Loc, Random), i.e. w_c = 1 for defects, 0 for 'none'.

Usage: python3 driver_sweep.py <out_dir>
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
import torch  # noqa: E402
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

CNN_DIR = "/Users/rtrevinnoc/maestria/tesis/research/nn/cnn"
sys.path.insert(0, CNN_DIR)
os.chdir(CNN_DIR)
sys.path.append(os.path.abspath(os.path.join(CNN_DIR, "../../simulation")))
import litho_sim  # noqa: E402
from model import WaferCNN  # noqa: E402

LABELS = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
          'Loc', 'Near-full', 'Random', 'Scratch']
PLAUSIBLE_DEFECTS = ['Scratch', 'Edge-Loc', 'Random']
# The 7 nm spec is unreachable in Case 1 (no fine stages). To study relative
# trajectory effects the threshold is scaled to the baseline config's median
# exposure MSD, computed at runtime from the first (baseline) run.
MSD_THRESHOLD = None
DIE_AREA = 1024.0

die_l = np.sqrt(DIE_AREA) * 1e-3
wafer_r = 0.150
alpha = 0.25
dies = litho_sim.get_wafer_dies(wafer_r, die_l)

cnn = WaferCNN(num_classes=9)
cnn.load_state_dict(torch.load(os.path.join(CNN_DIR, "wafer_cnn.pth"),
                               map_location="cpu"))
cnn.eval()

cmap = ListedColormap(["#f0f0f0", "#3a7ca5", "#e8c547"])


def moving_std(x, w):
    if len(x) < w:
        return np.array([np.std(x)])
    ma = np.convolve(x, np.ones(w) / w, mode="valid")
    ma2 = np.convolve(np.asarray(x) ** 2, np.ones(w) / w, mode="valid")
    return np.sqrt(np.maximum(ma2 - ma ** 2, 0.0))


def run_config(v, a, j, s, tag):
    controller = litho_sim.StepperController(dies, die_l, v, a, j, s, alpha)
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

    t_exp = 5.5e-3 / (v / alpha)
    window = max(1, int(t_exp / model.opt.timestep))
    max_sim_time = len(dies) * (controller.profile.t_total + controller.t_step + 0.1)

    die_msd = []
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
        # pre-step pairing: body state and reference both at time t
        if controller.state == "SCANNING":
            e_syn = ((data.body("wafer").xpos[1] - yw)
                     - alpha * (data.body("mask").xpos[1] - yr))
            cur.append(e_syn)
        cancel_reactions(data)
        mujoco.mj_step(model, data)
        if controller.die_idx != last_die_idx and last_die_idx != -1:
            die_msd.append(moving_std(np.array(cur), window).max() if cur else 0.0)
            cur = []
            if len(die_msd) >= len(dies):
                break
        last_die_idx = controller.die_idx

    die_msd = np.array(die_msd)
    global MSD_THRESHOLD
    if MSD_THRESHOLD is None:          # first run = baseline defines it
        MSD_THRESHOLD = float(np.median(die_msd))
        print(f"threshold = baseline median MSD = "
              f"{MSD_THRESHOLD*1e9:,.2f} nm", flush=True)
    status = np.where(die_msd > MSD_THRESHOLD, 2, 1)

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

    resized = cv2.resize(wm.astype(np.float32), (64, 64),
                         interpolation=cv2.INTER_NEAREST)
    with torch.no_grad():
        probs = torch.softmax(cnn(torch.from_numpy(resized)[None, None]), dim=1)[0]
    probs = probs.numpy()
    cost = float(sum(probs[LABELS.index(c)] for c in PLAUSIBLE_DEFECTS))
    pred = LABELS[int(probs.argmax())]

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(wm, cmap=cmap, vmin=0, vmax=2, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(out_dir, f"sweep_map_{tag}.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[{tag}] v={v} a={a} j={j} s={s:g} | scan_profile={controller.profile.t_total*1e3:.1f}ms "
          f"| MSD[nm]: median={np.median(die_msd)*1e9:.2f} max={die_msd.max()*1e9:.2f} "
          f"| failing={int((status == 2).sum())}/{len(dies)} | pred={pred} "
          f"| P(none)={probs[0]:.3f} cost={cost:.4f}", flush=True)
    top = np.argsort(probs)[::-1][:3]
    print("        top classes: " +
          ", ".join(f"{LABELS[k]}={probs[k]:.3f}" for k in top), flush=True)
    return die_msd


CONFIGS = [
    (0.8, 20.0, 1600.0, 1e5, "baseline"),    # must run first: sets threshold
    (1.2, 40.0, 5000.0, 5e5, "aggressive"),
    (0.4, 20.0, 1600.0, 1e5, "half_v"),
    (0.8, 10.0, 1600.0, 1e5, "half_a"),
    (0.8, 20.0, 400.0, 1e5, "quarter_j"),
    (0.8, 20.0, 1600.0, 1e4, "tenth_s"),
    (0.2, 5.0, 200.0, 5e3, "conservative"),
]

for cfg in CONFIGS:
    run_config(*cfg)
