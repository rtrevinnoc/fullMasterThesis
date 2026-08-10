import mujoco
import numpy as np
import pandas as pd
import sys
import os
import tqdm

# Add simulation path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation")))
import litho_sim

def run_random_sim(v_scan, a_max, j_max, s_max, num_dies=5):
    """Runs a simulation and returns a dataframe of kinematic states and errors."""
    die_area = 1024.0
    die_l = np.sqrt(die_area) * 1e-3
    wafer_r = 0.150
    alpha = 0.25
    
    dies = litho_sim.get_wafer_dies(wafer_r, die_l)[:num_dies]
    controller = litho_sim.StepperController(dies, die_l, v_scan, a_max, j_max, s_max, alpha)
    
    die_grid_xml = "" # Minimal for data gen
    model_xml = litho_sim.get_model_xml(die_l, die_grid_xml)
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    cancel_reactions = litho_sim.make_reaction_canceller(model)
    
    records = []
    
    # Run simulation
    # Each die takes ~0.2s + scan time
    max_time = num_dies * (controller.profile.t_total + controller.t_step + 0.1)
    
    prev_ref = None
    while data.time < max_time:
        xw, yw, xr, yr = controller.get_ref(data.time)
        if prev_ref is not None:      # LS error-velocity damping references
            dtm = model.opt.timestep
            data.ctrl[18] = (xw - prev_ref[0]) / dtm
            data.ctrl[19] = (yw - prev_ref[1]) / dtm
            data.ctrl[20] = (xr - prev_ref[2]) / dtm
            data.ctrl[21] = (yr - prev_ref[3]) / dtm
        prev_ref = (xw, yw, xr, yr)

        # Get kinematics from controller (specifically for the Y axis scan)
        # We only care about errors during SCANNING state usually
        if controller.state == "SCANNING":
            dt = data.time - controller.t_start
            p, v, a, j, s = controller.profile.get_kinematics(dt)
            
            # Set actuators
            data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
            data.ctrl[9] = xr; data.ctrl[10] = yr; data.ctrl[11] = 0

            # Performance metrics (pre-step pairing: state and ref both at t)
            wafer_pos = data.body('wafer').xpos
            mask_pos = data.body('mask').xpos
            ew_y = wafer_pos[1] - yw
            er_y = mask_pos[1] - yr
            e_syn = ew_y - alpha * er_y

            cancel_reactions(data)
            mujoco.mj_step(model, data)

            records.append({
                'time': data.time,
                'v': v, 'a': a, 'j': j, 's': s,
                'e_syn': e_syn
            })
        else:
            # Just step without recording
            data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
            data.ctrl[9] = xr; data.ctrl[10] = yr; data.ctrl[11] = 0
            cancel_reactions(data)
            mujoco.mj_step(model, data)
        
        if controller.die_idx >= num_dies:
            break
            
    return pd.DataFrame(records)

def generate_dataset(n_sims=100):
    all_data = []
    
    print(f"Generating data from {n_sims} simulations...")
    for i in tqdm.tqdm(range(n_sims)):
        # Balanced randomization: 50% slow/precision, 50% fast/standard
        if np.random.rand() > 0.5:
            v = np.random.uniform(0.01, 0.2)
            a = np.random.uniform(0.1, 5.0)
            j = np.random.uniform(10.0, 500.0)
            s = np.random.uniform(100.0, 1e4)
        else:
            v = np.random.uniform(0.2, 1.0)
            a = np.random.uniform(5.0, 40.0)
            j = np.random.uniform(500.0, 5000.0)
            s = np.random.uniform(1e4, 5e5)
        
        df = run_random_sim(v, a, j, s)
        df['sim_id'] = i
        all_data.append(df)
        
    full_df = pd.concat(all_data)
    save_path = "research/nn/lstm/sim_data.pkl"
    full_df.to_pickle(save_path)
    print(f"Dataset saved to {save_path}. Total samples: {len(full_df)}")

if __name__ == "__main__":
    generate_dataset()
