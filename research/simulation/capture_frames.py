import mujoco
import numpy as np
from PIL import Image
import os

# Import the model from the simulation script
import evidencia3_simulacion as sim

def capture_frames():
    model = mujoco.MjModel.from_xml_string(sim.MODEL_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    
    dies = sim.get_wafer_dies(sim.WAFER_R, sim.DIE_L)
    controller = sim.StepperController(dies)
    
    # Set a nice camera angle
    # Look at the wafer/reticle area
    scene_option = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    camera.distance = 1.5
    camera.azimuth = 90
    camera.elevation = -20
    camera.lookat = [0, 0, 0.65]

    captured_states = {"SCANNING": False, "STEPPING": False}
    
    print("Simulating to capture frames...")
    
    # Max simulation time to find states
    max_t = 5.0 
    dt = model.opt.timestep
    
    while data.time < max_t:
        xw, yw, xr, yr = controller.get_ref(data.time)
        
        # Set actuators (using indices from the script)
        data.ctrl[0] = xw; data.ctrl[1] = yw   # w_ls
        data.ctrl[6] = xr; data.ctrl[7] = yr   # r_ls
        
        mujoco.mj_step(model, data)
        
        # Capture SCANNING (middle of the first scan)
        if controller.state == "SCANNING" and not captured_states["SCANNING"] and data.time > 0.1:
            renderer.update_scene(data, camera)
            pixels = renderer.render()
            Image.fromarray(pixels).save("scanning_3d.png")
            captured_states["SCANNING"] = True
            print("Captured scanning_3d.png")
            
        # Capture STEPPING (middle of the first step)
        if controller.state == "STEPPING" and not captured_states["STEPPING"]:
            renderer.update_scene(data, camera)
            pixels = renderer.render()
            Image.fromarray(pixels).save("stepping_3d.png")
            captured_states["STEPPING"] = True
            print("Captured stepping_3d.png")
            
        if all(captured_states.values()):
            break

if __name__ == "__main__":
    capture_frames()
