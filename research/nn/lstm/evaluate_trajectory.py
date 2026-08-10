import torch
import numpy as np
import cv2
import pickle
import sys
import os
import argparse
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

def evaluate_with_lstm(v_scan, a_max, j_max, s_max, die_area=1024.0):
    # 1. Load Model and Scalers
    device = torch.device("cpu")
    model = WaferLSTM()
    model.load_state_dict(torch.load("research/nn/lstm/wafer_lstm.pth", map_location=device))
    model.eval()
    
    with open("research/nn/lstm/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        feature_scaler = scalers['feature']
        target_scaler = scalers['target']

    # 2. Setup Trajectory
    die_l = np.sqrt(die_area) * 1e-3
    wafer_r = 0.150
    alpha = 0.25
    dies = litho_sim.get_wafer_dies(wafer_r, die_l)
    
    # We'll simulate one full scan profile using the generator
    t_ramp = (v_scan/a_max) + (a_max/j_max)
    profile = litho_sim.Profile4thOrder(v_scan, a_max, j_max, s_max, die_l + v_scan * t_ramp)
    
    dt = 0.0001 # Match simulation timestep
    time_steps = np.arange(0, profile.t_total, dt)
    
    kinematics = []
    for t in time_steps:
        _, v, a, j, s = profile.get_kinematics(t)
        kinematics.append([v, a, j, s])
    
    kinematics = np.array(kinematics)
    
    # 3. Predict Errors
    features_scaled = feature_scaler.transform(kinematics)
    input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0) # (1, seq_len, 4)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).squeeze().numpy()
        
    pred_esyn = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    # 4. Calculate Metrics
    v_r_scan = v_scan / alpha
    t_exp = 5.5e-3 / v_r_scan
    window_size = int(t_exp / dt)
    if window_size < 1: window_size = 1
    
    ma, msd = calculate_ma_msd(pred_esyn, window_size)
    max_ma = np.max(np.abs(ma))
    max_msd = np.max(msd)
    
    # 5. Generate Wafer Map (assuming same result for every die for this trajectory)
    # This is a simplification: we're evaluating the TRAJECTORY itself.
    MA_THRESHOLD = 1e-9
    MSD_THRESHOLD = 7e-9
    
    is_failing = (max_ma >= MA_THRESHOLD or max_msd >= MSD_THRESHOLD)
    
    print(f"\nNeural Evaluation for [v={v_scan:.2f}, a={a_max:.2f}, j={j_max:.2f}, s={s_max:.2e}]")
    print(f"Predicted Max |MA|: {max_ma*1e9:.3f} nm (Threshold: 1.0nm)")
    print(f"Predicted Max MSD: {max_msd*1e9:.3f} nm (Threshold: 7.0nm)")
    print(f"Result: {'FAIL' if is_failing else 'PASS'}")
    
    return is_failing

if __name__ == "__main__":
    # Test a few points
    evaluate_with_lstm(0.8, 20.0, 1600.0, 1e5)
    evaluate_with_lstm(0.1, 1.0, 100.0, 1e4)
    evaluate_with_lstm(0.01, 0.1, 1.0, 10.0)
