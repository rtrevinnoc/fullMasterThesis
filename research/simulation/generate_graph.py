import mujoco
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add simulation path to import StepperController and parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import litho_sim

def calculate_ma_msd(signal, window_size):
    if len(signal) < window_size:
        return np.array([np.mean(signal)]*len(signal)), np.array([np.std(signal)]*len(signal))
    
    ma = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    signal_sq = signal**2
    ma_sq = np.convolve(signal_sq, np.ones(window_size)/window_size, mode='same')
    msd = np.sqrt(np.maximum(ma_sq - ma**2, 0.0))
    return ma, msd

def main():
    v_scan = 1.0 
    a_max = 20.0
    j_max = 1600.0
    s_max = 1e5
    die_area = 1024.0
    die_l = np.sqrt(die_area) * 1e-3
    alpha = 0.25
    
    dies = [(0, 0)]
    controller = litho_sim.StepperController(dies, die_l, v_scan, a_max, j_max, s_max, alpha)
    
    # Generate model with finer timestep for stability and precision
    xml = litho_sim.get_model_xml(die_l, "")
    xml = xml.replace('timestep="0.0001"', 'timestep="0.00001"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # PRE-INITIALIZATION: Set joints to the start-of-scan position to eliminate the 60mm jump bug
    xw0, yw0, xr0, yr0 = controller.get_ref(0.0)
    data.qpos[model.joint('w_ls_x').qposadr[0]] = xw0
    data.qpos[model.joint('w_ls_y').qposadr[0]] = yw0
    data.qpos[model.joint('r_ls_x').qposadr[0]] = xr0
    data.qpos[model.joint('r_ls_y').qposadr[0]] = yr0
    mujoco.mj_forward(model, data)
    
    times = []
    e_syns = []
    
    # Warm-up / Settling period (20ms)
    for _ in range(2000):
        xw, yw, xr, yr = controller.get_ref(data.time)
        data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[9] = xr; data.ctrl[10] = yr
        mujoco.mj_step(model, data)

    # Simulation Scan
    max_time = controller.profile.t_total + 0.1
    while data.time < max_time:
        xw, yw, xr, yr = controller.get_ref(data.time)
        data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
        data.ctrl[9] = xr; data.ctrl[10] = yr; data.ctrl[11] = 0
        
        # Open-loop (Pure Feedforward)
        data.ctrl[7] = 0; data.ctrl[16] = 0
        
        mujoco.mj_step(model, data)
        
        if controller.state == "SCANNING":
            wafer_pos = data.body('wafer').xpos
            mask_pos = data.body('mask').xpos
            # Calculate Synchronization Error (Overlay Proxy)
            e_syn = (wafer_pos[1] - yw) - alpha * (mask_pos[1] - yr)
            times.append(data.time)
            e_syns.append(e_syn)
    
    times = np.array(times)
    e_syns = np.array(e_syns) * 1e9 # Convert to nm
    
    v_r_scan = v_scan / alpha
    t_exp = 5.5e-3 / v_r_scan
    window_size = int(t_exp / model.opt.timestep)
    
    ma, msd = calculate_ma_msd(e_syns, window_size)
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, e_syns, label='Error $e_{syn}$', color='gray', alpha=0.3, linewidth=0.5)
    plt.plot(times, ma, label='MA (Moving Average)', color='blue', linewidth=1.5)
    plt.plot(times, msd, label='MSD (Moving Std Dev)', color='red', linewidth=1.5)
    
    plt.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Límite MA (1nm)')
    plt.axhline(y=-1.0, color='blue', linestyle='--')
    plt.axhline(y=7.0, color='red', linestyle='--', alpha=0.5, label='Límite MSD (7nm)')
    
    # Zoom into the scanning window to reveal nanometer-scale vibrations
    # plt.ylim([-80, 80]) 
    
    plt.title(f'Error de Sincronización en Lazo Abierto (Feedforward) @ {v_scan} m/s')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Error [nm]')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    
    output_path = "presentation/pictures/sim_graph_ff.png"
    plt.savefig(output_path, dpi=300)
    print(f"Corrected graph saved to {output_path}")

if __name__ == "__main__":
    main()
