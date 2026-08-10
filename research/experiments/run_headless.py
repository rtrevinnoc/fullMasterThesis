"""Run a research script headless: stub out glfw/mujoco.viewer, which abort
(libffi trampoline assertion) when imported without window-server access.
Usage: python3 run_headless.py <script.py> [args...]
"""
import sys
import types
import runpy

glfw_stub = types.ModuleType("glfw")
sys.modules["glfw"] = glfw_stub

import mujoco  # noqa: E402

viewer_stub = types.ModuleType("mujoco.viewer")


def _no_viewer(*args, **kwargs):
    raise RuntimeError("mujoco.viewer is stubbed out in headless mode")


viewer_stub.launch = _no_viewer
viewer_stub.launch_passive = _no_viewer
sys.modules["mujoco.viewer"] = viewer_stub
mujoco.viewer = viewer_stub

import os  # noqa: E402

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
