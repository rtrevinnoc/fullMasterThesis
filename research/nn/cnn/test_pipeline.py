import mujoco
import numpy as np
import torch
import cv2
from model import WaferCNN
import sys
import os

# Add simulation path to import StepperController and parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation")))
import litho_sim

def generate_simulation_wafer_map(v_scan=0.8, a_max=20.0, j_max=1600.0, s_max=1e5, die_area=1024.0, perturbation_type=None):
    """
    Runs a headless MuJoCo simulation and generates a wafer map (2D array).
    perturbation_type: 'vibration', 'drift', or None
    """
    # 1. Setup Parameters (matching litho_sim.py logic)
    die_l = np.sqrt(die_area) * 1e-3
    wafer_r = 0.150
    alpha = 0.25
    
    dies = litho_sim.get_wafer_dies(wafer_r, die_l)
    controller = litho_sim.StepperController(dies, die_l, v_scan, a_max, j_max, s_max, alpha)
    
    # Generate XML (we need a local version of MODEL_XML)
    # For simplicity, we'll use a simplified version of the XML construction from litho_sim.py
    # or just call a modified version of it.
    
    # Instead of rebuilding XML, let's just use the logic to simulate errors.
    # To truly "test", we should ideally use the MjModel.
    
    # Let's mock the simulation results for a specific trajectory if MuJoCo headless is complex,
    # OR we can actually run it since we have the tools.
    
    # Let's try to run it. We need the die_grid_xml.
    die_grid_xml = ""
    for cx, cy in dies:
        die_grid_xml += f'<geom type="box" size="{die_l/2 - 0.0005} {die_l/2 - 0.0005} 0.0011" pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'
    
    # Minimal XML for error tracking
    xml = litho_sim.get_model_xml(die_l, die_grid_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    die_errors = [] # List of (die_idx, max_error)
    
    print(f"Simulating {len(dies)} dies with perturbation: {perturbation_type}")
    
    # Define error threshold (e.g., 50nm for failure)
    FAIL_THRESHOLD = 50e-9 
    
    last_die_idx = -1
    current_die_max_error = 0
    
    while data.time < (len(dies) * 0.5): # Rough limit
        xw, yw, xr, yr = controller.get_ref(data.time)
        
        # Inject Perturbation
        if perturbation_type == 'vibration':
            # Add noise to actuators
            xw += np.random.normal(0, 50e-9) # Increased for better visibility in CNN
            yw += np.random.normal(0, 50e-9)
        elif perturbation_type == 'drift':
            # Systematic drift towards the edge
            dist_from_center = np.sqrt(xw**2 + yw**2)
            if dist_from_center > 0.1: # Edge effect
                yw += 1000e-9 # MUCH larger drift
        elif perturbation_type == 'scratch':
            # Vertical line of failure
            if abs(xw - 0.05) < 0.005:
                yw += 2000e-9 # MUCH larger scratch

        # Set actuators
        data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
        data.ctrl[9] = xr; data.ctrl[10] = yr; data.ctrl[11] = 0
        
        mujoco.mj_step(model, data)
        
        if controller.state == "SCANNING":
            # Calculate current error
            # We compare the WORLD position of the wafer with the TARGET
            wafer_xpos = data.body('wafer').xpos
            
            error = np.sqrt((wafer_xpos[0] - xw)**2 + (wafer_xpos[1] - yw)**2)
            current_die_max_error = max(current_die_max_error, error)
            
        if controller.die_idx != last_die_idx and last_die_idx != -1:
            # Finished a die
            die_errors.append(current_die_max_error)
            current_die_max_error = 0
            if len(die_errors) >= len(dies): break
            
        last_die_idx = controller.die_idx

    # print(f"Die errors (first 5): {die_errors[:5]}")
    # print(f"Mean error: {np.mean(die_errors):.2e}, Max error: {np.max(die_errors):.2e}")
    
    # Define error threshold for failure
    # Based on observation, even without perturbation there might be some lag.
    # We want to detect the INJECTED errors.
    FAIL_THRESHOLD = max(die_errors) * 1.01 if perturbation_type is None else 100e-9 # 100nm
    
    # If we want a fixed one for all:
    FAIL_THRESHOLD = 200e-9 # 200nm is a safe bet for this simulation lag


    # 3. Create the Wafer Map Grid
    # We need to map dies back to their i, j indices
    n = int(np.ceil(2 * wafer_r / die_l)) + 2
    wafer_map = np.zeros((2*n, 2*n), dtype=int)
    
    # Re-generate the grid to find mapping
    for i in range(-n, n):
        for j in range(-n, n):
            cx = i * die_l
            cy = j * die_l
            corners = [(cx-die_l/2, cy-die_l/2), (cx+die_l/2, cy-die_l/2),
                       (cx-die_l/2, cy+die_l/2), (cx+die_l/2, cy+die_l/2)]
            if all(np.sqrt(x**2 + y**2) < wafer_r for x, y in corners):
                # This is a valid die. Find its index in the 'dies' list
                # (which was sorted in a snake pattern)
                try:
                    idx = dies.index((cx, cy))
                    if idx < len(die_errors):
                        status = 2 if die_errors[idx] > FAIL_THRESHOLD else 1
                    else:
                        status = 1
                    wafer_map[j+n, i+n] = status
                except ValueError:
                    pass
    
    print(f"Total failing dies: {np.sum(wafer_map == 2)}")
    return wafer_map

def test_inference(wafer_map):
    # Load Model
    model = WaferCNN(num_classes=9)
    model_path = os.path.join(os.path.dirname(__file__), "wafer_cnn.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Preprocess
    # 1. Resize to 64x64
    resized = cv2.resize(wafer_map.astype(np.float32), (64, 64), interpolation=cv2.INTER_NEAREST)
    # 2. To Tensor
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
    
    labels = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']
    return labels[pred_idx], probabilities[0][pred_idx].item()

if __name__ == "__main__":
    # Test 0: None -> none?
    map_none = generate_simulation_wafer_map(perturbation_type=None)
    pred, conf = test_inference(map_none)
    print(f"Simulation (None) -> Predicted: {pred} ({conf:.2%})")

    # Test 1: Vibration -> Random?
    map_vib = generate_simulation_wafer_map(perturbation_type='vibration')
    pred, conf = test_inference(map_vib)
    print(f"Simulation (Vibration) -> Predicted: {pred} ({conf:.2%})")

    # Test 2: Drift -> Edge-Ring/Edge-Loc?
    map_drift = generate_simulation_wafer_map(perturbation_type='drift')
    pred, conf = test_inference(map_drift)
    print(f"Simulation (Drift) -> Predicted: {pred} ({conf:.2%})")
    
    # Test 3: Scratch -> Scratch?
    map_scratch = generate_simulation_wafer_map(perturbation_type='scratch')
    pred, conf = test_inference(map_scratch)
    print(f"Simulation (Scratch) -> Predicted: {pred} ({conf:.2%})")
