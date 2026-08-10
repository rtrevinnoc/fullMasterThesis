import torch
import numpy as np
import cv2
import pickle
import sys
import os
from model import WaferLSTM

# Add simulation path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation")))
import litho_sim

def calculate_ma_msd(signal, window_size):
    if len(signal) < window_size:
        return np.mean(signal), np.std(signal)
    ma = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    signal_sq = signal**2
    ma_sq = np.convolve(signal_sq, np.ones(window_size)/window_size, mode='valid')
    msd = np.sqrt(np.maximum(ma_sq - ma**2, 0.0))
    return ma, msd

def generate_neural_wafer_map(v_scan, a_max, j_max, s_max, die_area=1024.0):
    # 1. Setup Parameters
    die_l = np.sqrt(die_area) * 1e-3
    wafer_r = 0.150
    alpha = 0.25
    dies = litho_sim.get_wafer_dies(wafer_r, die_l)
    
    # 2. Load LSTM
    device = torch.device("cpu")
    model = WaferLSTM()
    model.load_state_dict(torch.load("research/nn/lstm/wafer_lstm.pth", map_location=device))
    model.eval()
    with open("research/nn/lstm/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        feature_scaler = scalers['feature']
        target_scaler = scalers['target']

    # 3. Simulate one scan profile
    t_ramp = (v_scan/a_max) + (a_max/j_max)
    profile = litho_sim.Profile4thOrder(v_scan, a_max, j_max, s_max, die_l + v_scan * t_ramp)
    dt = 0.0001
    time_steps = np.arange(0, profile.t_total, dt)
    kinematics = []
    for t in time_steps:
        _, v, a, j, s = profile.get_kinematics(t)
        kinematics.append([v, a, j, s])
    
    # 4. Predict
    features_scaled = feature_scaler.transform(kinematics)
    input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(input_tensor).squeeze().numpy()
    pred_esyn = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    # 5. Metrics
    v_r_scan = v_scan / alpha
    t_exp = 5.5e-3 / v_r_scan
    window_size = int(t_exp / dt)
    if window_size < 1: window_size = 1
    ma, msd = calculate_ma_msd(pred_esyn, window_size)
    
    # Check if this trajectory FAILS globally
    # (Since we simulate only one scan, we apply it to all dies)
    # This is where the LSTM "evaluates" the trajectory.
    MA_THRESHOLD = 1e-9
    MSD_THRESHOLD = 7e-9
    status = 2 if (np.max(np.abs(ma)) >= MA_THRESHOLD or np.max(msd) >= MSD_THRESHOLD) else 1
    
    # 6. Build Grid
    n = int(np.ceil(2 * wafer_r / die_l)) + 2
    wafer_map = np.zeros((2*n, 2*n), dtype=int)
    for i in range(-n, n):
        for j in range(-n, n):
            cx = i * die_l
            cy = j * die_l
            corners = [(cx-die_l/2, cy-die_l/2), (cx+die_l/2, cy-die_l/2),
                       (cx-die_l/2, cy+die_l/2), (cx+die_l/2, cy+die_l/2)]
            if all(np.sqrt(x**2 + y**2) < wafer_r for x, y in corners):
                wafer_map[j+n, i+n] = status
                
    return wafer_map

if __name__ == "__main__":
    # Example: Neural prediction of a map
    wm = generate_neural_wafer_map(0.8, 20.0, 1600.0, 1e5)
    print(f"Neural Wafer Map Generated. Total failing dies (predicted): {np.sum(wm == 2)}")
    
    # Save the map to test against the CNN
    np.save("research/nn/lstm/neural_wafer_map.npy", wm)
