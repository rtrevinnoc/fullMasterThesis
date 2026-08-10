import mujoco
import numpy as np
import torch
import cv2
from model import WaferCNN
import sys
import os
import argparse

# Add simulation path to import StepperController and parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation")))
import litho_sim

def calculate_ma_msd(signal, window_size):
    """
    Calculates Moving Average and Moving Standard Deviation.
    """
    if len(signal) < window_size:
        return np.mean(signal), np.std(signal)
    
    ma = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    signal_sq = signal**2
    ma_sq = np.convolve(signal_sq, np.ones(window_size)/window_size, mode='valid')
    msd = np.sqrt(np.maximum(ma_sq - ma**2, 0.0))
    return ma, msd

def generate_dynamic_wafer_map(v_scan, a_max, j_max, s_max, die_area, control_mode='ff'):
    """
    Generates a wafer map based on MA/MSD thresholds of synchronization error.
    """
    die_l = np.sqrt(die_area) * 1e-3
    wafer_r = 0.150; alpha = 0.25
    v_r_scan = v_scan / alpha
    t_exp = 5.5e-3 / v_r_scan
    
    dies = litho_sim.get_wafer_dies(wafer_r, die_l)
    controller = litho_sim.StepperController(dies, die_l, v_scan, a_max, j_max, s_max, alpha)
    
    die_grid_xml = ""
    for cx, cy in dies:
        die_grid_xml += f'<geom type="box" size="{die_l/2 - 0.0005} {die_l/2 - 0.0005} 0.0011" pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'
    
    xml = litho_sim.get_model_xml(die_l, die_grid_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # Initialize HIGS if requested
    higs_w = litho_sim.HIGSController(model.opt.timestep, 2e6, 2*np.pi*34.4, 2*np.pi*20, 1.0)
    higs_r = litho_sim.HIGSController(model.opt.timestep, 2e6, 2*np.pi*34.4, 2*np.pi*20, 1.0)
    
    MA_THRESHOLD = 1e-9
    MSD_THRESHOLD = 7e-9
    window_size = int(t_exp / model.opt.timestep)
    if window_size < 1: window_size = 1

    die_status = []
    print(f"Simulating {len(dies)} dies [v={v_scan}, a={a_max}, mode={control_mode}]")

    current_die_esyn = []
    last_die_idx = -1
    max_sim_time = len(dies) * (controller.profile.t_total + controller.t_step + 0.1)
    
    while data.time < max_sim_time:
        xw, yw, xr, yr = controller.get_ref(data.time)
        data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
        data.ctrl[9] = xr; data.ctrl[10] = yr; data.ctrl[11] = 0
        
        if control_mode == 'higs':
            ew_y = yw - data.body('wafer').xpos[1]
            er_y = yr - data.body('mask').xpos[1]
            ew_y_dot = -data.qvel[model.joint('wafer_y').dofadr[0]]
            er_y_dot = -data.qvel[model.joint('mask_y').dofadr[0]]
            data.ctrl[7] = higs_w.update(ew_y, ew_y_dot)
            data.ctrl[16] = higs_r.update(er_y, er_y_dot)
            
        mujoco.mj_step(model, data)
        
        if controller.state == "SCANNING":
            wafer_pos = data.body('wafer').xpos
            mask_pos = data.body('mask').xpos
            e_syn = (wafer_pos[1] - yw) - alpha * (mask_pos[1] - yr)
            current_die_esyn.append(e_syn)
            
        if controller.die_idx != last_die_idx:
            if last_die_idx != -1:
                if len(current_die_esyn) > 0:
                    ma, msd = calculate_ma_msd(np.array(current_die_esyn), window_size)
                    status = 2 if (np.max(np.abs(ma)) >= MA_THRESHOLD or np.max(msd) >= MSD_THRESHOLD) else 1
                    die_status.append(status)
                else: die_status.append(1)
                current_die_esyn = []
            if len(die_status) >= len(dies): break
            last_die_idx = controller.die_idx

    n = int(np.ceil(2 * wafer_r / die_l)) + 2
    wafer_map = np.zeros((2*n, 2*n), dtype=int)
    for i in range(-n, n):
        for j in range(-n, n):
            cx = i * die_l; cy = j * die_l
            corners = [(cx-die_l/2, cy-die_l/2), (cx+die_l/2, cy-die_l/2),
                       (cx-die_l/2, cy+die_l/2), (cx+die_l/2, cy+die_l/2)]
            if all(np.sqrt(x**2 + y**2) < wafer_r for x, y in corners):
                try:
                    idx = dies.index((cx, cy))
                    if idx < len(die_status): wafer_map[j+n, i+n] = die_status[idx]
                except ValueError: pass
    
    print(f"Failing dies: {np.sum(wafer_map == 2)} / {len(dies)}")
    return wafer_map

def main():
    parser = argparse.ArgumentParser(description="Test Wafer Trajectory and Classify Defects")
    parser.add_argument("--v-scan", type=float, default=0.8)
    parser.add_argument("--a-max", type=float, default=20.0)
    parser.add_argument("--j-max", type=float, default=1600.0)
    parser.add_argument("--s-max", type=float, default=1e5)
    parser.add_argument("--die-area", type=float, default=1024.0)
    parser.add_argument("--control", choices=['ff', 'higs'], default='ff')
    args = parser.parse_args()

    wafer_map = generate_dynamic_wafer_map(args.v_scan, args.a_max, args.j_max, args.s_max, args.die_area, args.control)
    
    model = WaferCNN(num_classes=9)
    model_path = os.path.join(os.path.dirname(__file__), "wafer_cnn.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    resized = cv2.resize(wafer_map.astype(np.float32), (64, 64), interpolation=cv2.INTER_NEAREST)
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
    
    labels = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']
    print(f"\nFinal Result -> Predicted Failure Pattern: {labels[pred_idx]} ({probabilities[0][pred_idx].item():.2%})")

if __name__ == "__main__":
    main()
