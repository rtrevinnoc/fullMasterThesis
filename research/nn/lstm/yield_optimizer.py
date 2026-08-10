import torch
import numpy as np
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

def predict_performance(model, feature_scaler, target_scaler, v, a, j, s):
    # Setup Trajectory
    die_l = 0.032 # 1024 mm^2
    alpha = 0.25
    t_ramp = (v/a) + (a/j)
    profile = litho_sim.Profile4thOrder(v, a, j, s, die_l + v * t_ramp)
    
    dt = 0.0001
    time_steps = np.arange(0, profile.t_total, dt)
    kinematics = []
    for t in time_steps:
        _, kv, ka, kj, ks = profile.get_kinematics(t)
        kinematics.append([kv, ka, kj, ks])
    
    kinematics = np.array(kinematics)
    features_scaled = feature_scaler.transform(kinematics)
    input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).squeeze().numpy()
    
    # Handle single point edge case
    if pred_scaled.ndim == 0: pred_scaled = np.array([pred_scaled])
        
    pred_esyn = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    t_exp = 5.5e-3 / (v / alpha)
    window_size = int(t_exp / dt)
    if window_size < 1: window_size = 1
    
    ma, msd = calculate_ma_msd(pred_esyn, window_size)
    return np.max(np.abs(ma)), np.max(msd)

def optimize_yield():
    # 1. Load Surrogate Stack
    device = torch.device("cpu")
    model = WaferLSTM()
    model.load_state_dict(torch.load("research/nn/lstm/wafer_lstm.pth", map_location=device))
    model.eval()
    with open("research/nn/lstm/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        feature_scaler = scalers['feature']
        target_scaler = scalers['target']

    print("Searching for high-yield snap-bounded parameters...")
    
    # We want to find the highest v that passes
    # Grid search for simplicity and robustness
    best_v = 0
    best_params = None
    
    # Search space
    vs = np.linspace(0.01, 0.5, 10)
    as_ = np.linspace(0.1, 10.0, 5)
    js = [500, 1000, 2000]
    ss = [1e4, 5e4, 1e5]
    
    for v in vs:
        found_pass_for_v = False
        for a in as_:
            for j in js:
                for s in ss:
                    ma, msd = predict_performance(model, feature_scaler, target_scaler, v, a, j, s)
                    
                    # Yield criteria
                    if ma < 1e-9 and msd < 7e-9:
                        if v > best_v:
                            best_v = v
                            best_params = (v, a, j, s, ma, msd)
                        found_pass_for_v = True
        
        if not found_pass_for_v and v > 0.1:
            # If we can't find a pass at this speed, higher speeds likely won't either
            # (In Case 1, the limit is very low)
            pass

    if best_params:
        v, a, j, s, ma, msd = best_params
        print(f"\n--- OPTIMIZED YIELD TRAJECTORY FOUND ---")
        print(f"Speed (v): {v:.3f} m/s")
        print(f"Accel (a): {a:.2f} m/s^2")
        print(f"Jerk  (j): {j:.1f} m/s^3")
        print(f"Snap  (s): {s:.1e} m/s^4")
        print(f"Predicted Precision: MA={ma*1e9:.3f}nm, MSD={msd*1e9:.3f}nm")
        print(f"Result: SUCCESS (Yield Increase)")
        return best_params
    else:
        print("\nNo trajectory found within the sub-nanometer limit for the current machine (Case 1).")
        print("Consider upgrading to Case 2 (Spring Compensation) or Case 3 (Fine Stages).")
        return None

if __name__ == "__main__":
    optimize_yield()
