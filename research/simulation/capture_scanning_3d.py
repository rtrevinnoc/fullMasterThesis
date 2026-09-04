"""Regenerate scanning_3d.png from the current two-stage MuJoCo model
(litho_sim.py), replacing the stale screenshot inherited from the old
"evidencia 3" coursework model (research/simulation/evidencia3_simulacion.py,
which predates the thesis's LS+SS/passive-Stage machine).

Usage: python3 capture_scanning_3d.py <out_dir>
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
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import litho_sim as sim  # noqa: E402

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

die_l = np.sqrt(1024.0) * 1e-3
wafer_r = 0.150
alpha = 0.25
dies = sim.get_wafer_dies(wafer_r, die_l)
die_grid_xml = ""
for cx, cy in dies:
    die_grid_xml += (
        f'<geom type="box" size="{die_l/2 - 0.0005} {die_l/2 - 0.0005} 0.0011" '
        f'pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'
    )
model_xml = sim.get_model_xml(die_l, die_grid_xml)
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)
controller = sim.StepperController(dies, die_l, 0.8, 20.0, 1600.0, 1e5, alpha)
cancel_reactions = sim.make_reaction_canceller(model)

renderer = mujoco.Renderer(model, width=640, height=480)
camera = mujoco.MjvCamera()
camera.distance = 1.4
camera.azimuth = 120
camera.elevation = -22
camera.lookat = [0, 0, 0.65]

prev_ref = None
captured = False
while data.time < 5.0 and not captured:
    xw, yw, xr, yr = controller.get_ref(data.time)
    data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
    data.ctrl[6] = xr; data.ctrl[7] = yr; data.ctrl[8] = 0
    if prev_ref is not None:
        dt = model.opt.timestep
        data.ctrl[12] = (xw - prev_ref[0]) / dt
        data.ctrl[13] = (yw - prev_ref[1]) / dt
        data.ctrl[14] = (xr - prev_ref[2]) / dt
        data.ctrl[15] = (yr - prev_ref[3]) / dt
    prev_ref = (xw, yw, xr, yr)
    data.ctrl[4] = 0; data.ctrl[10] = 0
    cancel_reactions(data)
    mujoco.mj_step(model, data)

    if controller.state == "SCANNING" and data.time > 0.15:
        renderer.update_scene(data, camera)
        pixels = renderer.render()
        Image.fromarray(pixels).save(os.path.join(out_dir, "scanning_3d.png"))
        print(f"Captured scanning_3d.png at t={data.time:.3f}s, state={controller.state}")
        captured = True

if not captured:
    raise RuntimeError("Never reached SCANNING state before t=5.0s")
