import math
import numpy as np

# Read the current litho_sim.py just to extract the controller logic if needed, 
# but it's simpler to just write the whole thing out since I know it.

content = """import mujoco
import mujoco.viewer
import time
import numpy as np
import sys

# ============================================================
# PAPER PARAMETERS - Appendix B
# Al-Rawashdeh et al., Mech. Mach. Theory 170 (2022) 104638
# ============================================================

# --- Common Stage Masses (Reticle j=1, Wafer j=3) ---
M_BASE = 1400.0  # m1
M_ACT  = 0.5     # m2, m4, m6
M_LS_F = 49.0    # m3
M_SS_F = 28.0    # m5
M_ST_F = 10.5    # m7
M_MASK = 0.04    # m8 reticle
M_WAFR = 0.2     # m8 wafer

# --- Optics Chain (j=2) ---
M_METRO = 520.0  # m2
M_OPT_B = 200.0  # m3
M_LENS  = 0.3    # m4

# --- Stiffness (K) and Damping (C) ---
# Wafer Chain End-Effector (j=3, m8)
K8_W = 0.03158 * 1e9
C8_W = 0.0000355 * 1e8

# Reticle Chain End-Effector (j=1, m8)
K8_R = 0.0063165 * 1e9
C8_R = 710.0

# General Flex stages (j=1,3: m3, m5, m7)
K_LS_F = 0.01398 * 1e9; C_LS_F = 0.0003701 * 1e8
K_SS_F = 0.01462 * 1e9; C_SS_F = 0.0002861 * 1e8
K_ST_F = 0.01343 * 1e9; C_ST_F = 0.0001679 * 1e8

# Base-Floor (K1, C1)
K1 = 0.00553 * 1e9; C1 = 0.001244 * 1e8

# Optics Chain (j=2)
K2_O = 0.4619 * 1e9; C2_O = 0.0069308 * 1e8
K3_O = 6.396e7;      C3_O = 1.5994e5
K4_O = 1.07e6;       C4_O = 800.0

# ============================================================
# LITHOGRAPHY SPECIFICATIONS
# ============================================================
DIE_AREA = 150.0 # mm^2
DIE_L = np.sqrt(DIE_AREA) * 1e-3 # m (~12.25 mm)
WAFER_R = 0.150 # m (300mm wafer)
ALPHA = 0.25 # Demagnification (Reticle is 4x larger/faster)

V_W_SCAN = 0.8 # m/s
V_R_SCAN = V_W_SCAN / ALPHA # 3.2 m/s
A_W_MAX  = 20.0 # m/s^2
A_R_MAX  = A_W_MAX / ALPHA # 80 m/s^2

def get_wafer_dies(radius, die_l):
    dies = []
    n = int(np.ceil(2 * radius / die_l)) + 2
    for i in range(-n, n):
        for j in range(-n, n):
            cx = i * die_l
            cy = j * die_l
            corners = [(cx-die_l/2, cy-die_l/2), (cx+die_l/2, cy-die_l/2),
                       (cx-die_l/2, cy+die_l/2), (cx+die_l/2, cy+die_l/2)]
            if all(np.sqrt(x**2 + y**2) < radius for x, y in corners):
                dies.append((cx, cy))
    
    cols = {}
    for x, y in dies:
        if x not in cols: cols[x] = []
        cols[x].append(y)
    
    sorted_dies = []
    x_coords = sorted(cols.keys())
    for i, x in enumerate(x_coords):
        y_coords = sorted(cols[x])
        if i % 2 == 1: y_coords = y_coords[::-1]
        for y in y_coords:
            sorted_dies.append((x, y))
    return sorted_dies

die_grid_xml = ""
for cx, cy in get_wafer_dies(WAFER_R, DIE_L):
    die_grid_xml += f'                                        <geom type="box" size="{DIE_L/2 - 0.0005} {DIE_L/2 - 0.0005} 0.0011" pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\\n'

MODEL_XML = f\"\"\"
<mujoco model="LITHO_FULL_MACHINE">
    <option integrator="implicit" timestep="0.0001" gravity="0 0 -9.81"/>

    <visual>
        <headlight ambient="0.3 0.3 0.3" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <map shadowclip="2" shadowscale="0.6"/>
        <rgba haze="0.15 0.15 0.18 1"/>
    </visual>

    <asset>
        <material name="granite"   rgba="0.15 0.15 0.17 1"   specular="0.3"/>
        <material name="metrology" rgba="0.4 0.4 0.45 1"     specular="0.5"/>
        <material name="ls_steel"  rgba="0.3 0.35 0.45 1"    specular="0.6"/>
        <material name="ss_alum"   rgba="0.6 0.6 0.65 1"     specular="0.8"/>
        <material name="ws_alum"   rgba="0.7 0.7 0.75 1"     specular="0.9"/>
        <material name="silicon"   rgba="0.2 0.2 0.3 0.9"    specular="1.0" shininess="1"/>
        <material name="glass"     rgba="0.8 0.9 1.0 0.3"    specular="1.0"/>
        <material name="floor"     rgba="0.5 0.5 0.5 1"/>
    </asset>

    <worldbody>
        <light pos="1 1 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom type="plane" size="5 5 0.1" material="floor"/>

        <!-- S1: Base Frame (m1=1400kg) -->
        <body name="base_frame" pos="0 0 0.2">
            <joint name="base_x" type="slide" axis="1 0 0" stiffness="{K1}" damping="{C1}"/>
            <joint name="base_y" type="slide" axis="0 1 0" stiffness="{K1}" damping="{C1}"/>
            <joint name="base_z" type="slide" axis="0 0 1" stiffness="{K1}" damping="{C1}"/>
            <geom type="box" size="1.2 1.0 0.2" material="granite" mass="{M_BASE}"/>

            <!-- j=2: Optics Chain (4 bodies total with base_frame) -->
            <!-- 1. Metrology Frame (m2=520kg) -->
            <body name="metro_frame" pos="0 0 0.5">
                <joint name="metro_x" type="slide" axis="1 0 0" stiffness="{K2_O}" damping="{C2_O}"/>
                <joint name="metro_y" type="slide" axis="0 1 0" stiffness="{K2_O}" damping="{C2_O}"/>
                <joint name="metro_z" type="slide" axis="0 0 1" stiffness="{K2_O}" damping="{C2_O}"/>
                <!-- Metrology Frame Structure (H-shape legs) -->
                <geom type="box" size="0.1 0.1 0.5" pos="0.8 0.6 0" material="metrology" mass="{M_METRO/4}"/>
                <geom type="box" size="0.1 0.1 0.5" pos="-0.8 0.6 0" material="metrology" mass="{M_METRO/4}"/>
                <geom type="box" size="0.1 0.1 0.5" pos="0.8 -0.6 0" material="metrology" mass="{M_METRO/4}"/>
                <geom type="box" size="0.1 0.1 0.5" pos="-0.8 -0.6 0" material="metrology" mass="{M_METRO/4}"/>
                <geom type="box" size="0.9 0.7 0.05" pos="0 0 0.5" material="metrology" density="1"/>

                <!-- 2. Optics Box (m3=200kg) -->
                <body name="optics_box" pos="0 0 0.3">
                    <joint name="optics_x" type="slide" axis="1 0 0" stiffness="{K3_O}" damping="{C3_O}"/>
                    <joint name="optics_y" type="slide" axis="0 1 0" stiffness="{K3_O}" damping="{C3_O}"/>
                    <joint name="optics_z" type="slide" axis="0 0 1" stiffness="{K3_O}" damping="{C3_O}"/>
                    <geom type="cylinder" size="0.2 0.3" material="metrology" mass="{M_OPT_B}"/>
                    
                    <!-- 3. Lens Element (m4=0.3kg) -->
                    <body name="lens" pos="0 0 -0.4">
                        <joint name="lens_z" type="slide" axis="0 0 1" stiffness="{K4_O}" damping="{C4_O}"/>
                        <geom type="cylinder" size="0.1 0.05" material="glass" mass="{M_LENS}"/>
                        <!-- Die indicator fixed to the optics chain -->
                        <geom name="die_indicator" type="box" size="{DIE_L/2} {DIE_L/2} 0.002" pos="0 0 0.015" rgba="1 1 0 0.6" contype="0" conaffinity="0"/>
                    </body>
                </body>
            </body>

            <!-- j=3: Wafer Chain (7 bodies) -->
            <!-- 1. LS Actuated (m2) -->
            <body name="w_ls_act" pos="0 0 0.25">
                <joint name="w_ls_x" type="slide" axis="1 0 0" stiffness="0" damping="500"/>
                <joint name="w_ls_y" type="slide" axis="0 1 0" stiffness="0" damping="500"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" mass="{M_ACT}"/>
                <!-- 2. LS Flex (m3) -->
                <body name="w_ls_flex" pos="0 0 0.03">
                    <joint name="w_lsf_x" type="slide" axis="1 0 0" stiffness="{K_LS_F}" damping="{C_LS_F}"/>
                    <joint name="w_lsf_y" type="slide" axis="0 1 0" stiffness="{K_LS_F}" damping="{C_LS_F}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" mass="{M_LS_F}"/>
                    <!-- 3. SS Actuated (m4) -->
                    <body name="w_ss_act" pos="0 0 0.04">
                        <joint name="w_ss_x" type="slide" axis="1 0 0" stiffness="0" damping="500"/>
                        <joint name="w_ss_y" type="slide" axis="0 1 0" stiffness="0" damping="500"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" mass="{M_ACT}"/>
                        <!-- 4. SS Flex (m5) -->
                        <body name="w_ss_flex" pos="0 0 0.03">
                            <joint name="w_ssf_x" type="slide" axis="1 0 0" stiffness="{K_SS_F}" damping="{C_SS_F}"/>
                            <joint name="w_ssf_y" type="slide" axis="0 1 0" stiffness="{K_SS_F}" damping="{C_SS_F}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" mass="{M_SS_F}"/>
                            <!-- 5. Fine Stage Actuated (m6) -->
                            <body name="w_st_act" pos="0 0 0.04">
                                <joint name="w_st_x" type="slide" axis="1 0 0" stiffness="0" damping="500"/>
                                <joint name="w_st_y" type="slide" axis="0 1 0" stiffness="0" damping="500"/>
                                <geom type="cylinder" size="0.16 0.015" material="ws_alum" mass="{M_ACT}"/>
                                <!-- 6. Fine Stage Flex (m7) -->
                                <body name="w_st_flex" pos="0 0 0.02">
                                    <joint name="w_stf_x" type="slide" axis="1 0 0" stiffness="{K_ST_F}" damping="{C_ST_F}"/>
                                    <joint name="w_stf_y" type="slide" axis="0 1 0" stiffness="{K_ST_F}" damping="{C_ST_F}"/>
                                    <geom type="cylinder" size="0.16 0.015" material="ws_alum" mass="{M_ST_F}"/>
                                    <!-- 7. End Effector (m8) -->
                                    <body name="wafer" pos="0 0 0.02">
                                        <joint name="wafer_x" type="slide" axis="1 0 0" stiffness="{K8_W}" damping="{C8_W}"/>
                                        <joint name="wafer_y" type="slide" axis="0 1 0" stiffness="{K8_W}" damping="{C8_W}"/>
                                        <geom type="cylinder" size="0.15 0.001" material="silicon" mass="{M_WAFR}"/>
{die_grid_xml}                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- j=1: Reticle Chain (7 bodies) -->
            <!-- 1. LS Actuated (m2) -->
            <body name="r_ls_act" pos="0 0 1.2">
                <joint name="r_ls_x" type="slide" axis="1 0 0" stiffness="0" damping="500"/>
                <joint name="r_ls_y" type="slide" axis="0 1 0" stiffness="0" damping="500"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" mass="{M_ACT}"/>
                <!-- 2. LS Flex (m3) -->
                <body name="r_ls_flex" pos="0 0 -0.03">
                    <joint name="r_lsf_x" type="slide" axis="1 0 0" stiffness="{K_LS_F}" damping="{C_LS_F}"/>
                    <joint name="r_lsf_y" type="slide" axis="0 1 0" stiffness="{K_LS_F}" damping="{C_LS_F}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" mass="{M_LS_F}"/>
                    <!-- 3. SS Actuated (m4) -->
                    <body name="r_ss_act" pos="0 0 -0.04">
                        <joint name="r_ss_x" type="slide" axis="1 0 0" stiffness="0" damping="500"/>
                        <joint name="r_ss_y" type="slide" axis="0 1 0" stiffness="0" damping="500"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" mass="{M_ACT}"/>
                        <!-- 4. SS Flex (m5) -->
                        <body name="r_ss_flex" pos="0 0 -0.03">
                            <joint name="r_ssf_x" type="slide" axis="1 0 0" stiffness="{K_SS_F}" damping="{C_SS_F}"/>
                            <joint name="r_ssf_y" type="slide" axis="0 1 0" stiffness="{K_SS_F}" damping="{C_SS_F}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" mass="{M_SS_F}"/>
                            <!-- 5. Fine Stage Actuated (m6) -->
                            <body name="r_st_act" pos="0 0 -0.04">
                                <joint name="r_st_x" type="slide" axis="1 0 0" stiffness="0" damping="500"/>
                                <joint name="r_st_y" type="slide" axis="0 1 0" stiffness="0" damping="500"/>
                                <geom type="box" size="0.1 0.1 0.015" material="ws_alum" mass="{M_ACT}"/>
                                <!-- 6. Fine Stage Flex (m7) -->
                                <body name="r_st_flex" pos="0 0 -0.02">
                                    <joint name="r_stf_x" type="slide" axis="1 0 0" stiffness="{K_ST_F}" damping="{C_ST_F}"/>
                                    <joint name="r_stf_y" type="slide" axis="0 1 0" stiffness="{K_ST_F}" damping="{C_ST_F}"/>
                                    <geom type="box" size="0.1 0.1 0.015" material="ws_alum" mass="{M_ST_F}"/>
                                    <!-- 7. End Effector (m8) -->
                                    <body name="mask" pos="0 0 -0.02">
                                        <joint name="mask_x" type="slide" axis="1 0 0" stiffness="{K8_R}" damping="{C8_R}"/>
                                        <joint name="mask_y" type="slide" axis="0 1 0" stiffness="{K8_R}" damping="{C8_R}"/>
                                        <geom type="box" size="0.05 0.05 0.005" material="glass" mass="{M_MASK}"/>
                                        <!-- Highlight the reticle exposure area -->
                                        <geom type="box" size="{DIE_L*2} {DIE_L*2} 0.006" pos="0 0 -0.005" rgba="1 1 0 0.2" contype="0" conaffinity="0"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <!-- Stationary exposure slit projected from the lens -->
        <body name="exposure_field" pos="0 0 0.72">
            <geom type="box" size="{DIE_L/2} {DIE_L/2} 0.001" rgba="0 1 1 0.3" contype="0" conaffinity="0"/>
            <geom type="box" size="{DIE_L/2} {DIE_L/2} 0.5" pos="0 0 0.25" rgba="0 1 1 0.05" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <actuator>
        <position name="wafer_ls_x" joint="w_ls_x" kp="1e7"/>
        <position name="wafer_ls_y" joint="w_ls_y" kp="1e7"/>
        <position name="wafer_ss_x" joint="w_ss_x" kp="1e7"/>
        <position name="wafer_ss_y" joint="w_ss_y" kp="1e7"/>
        <position name="wafer_st_x" joint="w_st_x" kp="1e7"/>
        <position name="wafer_st_y" joint="w_st_y" kp="1e7"/>
        
        <position name="reticle_ls_x" joint="r_ls_x" kp="1e7"/>
        <position name="reticle_ls_y" joint="r_ls_y" kp="1e7"/>
        <position name="reticle_ss_x" joint="r_ss_x" kp="1e7"/>
        <position name="reticle_ss_y" joint="r_ss_y" kp="1e7"/>
        <position name="reticle_st_x" joint="r_st_x" kp="1e7"/>
        <position name="reticle_st_y" joint="r_st_y" kp="1e7"/>
    </actuator>
</mujoco>
\"\"\"

class StepperController:
    def __init__(self, dies):
        self.dies = dies
        self.die_idx = 0
        self.state = "IDLE"
        self.t_start = 0
        
        self.scan_dist = DIE_L * 1.5
        self.v_scan = V_W_SCAN
        self.a_scan = A_W_MAX
        self.t_accel = self.v_scan / self.a_scan
        self.d_accel = 0.5 * self.a_scan * self.t_accel**2
        self.d_const = DIE_L
        self.t_const = self.d_const / self.v_scan
        
        self.t_total_scan = 2 * self.t_accel + self.t_const
        self.t_step = 0.2
        
    def get_ref(self, t):
        if self.die_idx >= len(self.dies):
            self.die_idx = 0
            self.state = "IDLE"
            self.t_start = t
            
        cx, cy = self.dies[self.die_idx]
        dt = t - self.t_start
        
        dir_y = 1 if (self.die_idx % 2 == 0) else -1
        
        if self.state == "IDLE":
            self.state = "SCANNING"
            self.t_start = t
            y_offset = -dir_y * (self.d_accel + self.d_const/2)
            return -cx, -cy + y_offset
            
        elif self.state == "SCANNING":
            if dt < self.t_accel:
                dy = 0.5 * self.a_scan * dt**2
            elif dt < self.t_accel + self.t_const:
                dy = self.d_accel + self.v_scan * (dt - self.t_accel)
            elif dt < self.t_total_scan:
                t_dec = dt - (self.t_accel + self.t_const)
                dy = self.d_accel + self.d_const + (self.v_scan * t_dec - 0.5 * self.a_scan * t_dec**2)
            else:
                self.state = "STEPPING"
                self.t_start = t
                dt = 0
                dy = 2 * self.d_accel + self.d_const
            
            y_ref = -cy + dir_y * (dy - (self.d_accel + self.d_const/2))
            return -cx, y_ref
            
        elif self.state == "STEPPING":
            if dt > self.t_step:
                self.die_idx += 1
                self.state = "SCANNING"
                self.t_start = t
            
            y_ref = -cy + dir_y * (self.d_accel + self.d_const/2)
            return -cx, y_ref

MONITOR_CONFIG = [
    ("base_x",  "X", "Base frame (m1) [Floor]", K1),
    ("base_y",  "Y", "Base frame (m1) [Floor]", K1),
    ("base_z",  "Z", "Base frame (m1) [Floor]", K1),
    
    ("metro_x", "X", "Metrology (j=2, m2) [Metro Frame]", K2_O),
    ("metro_y", "Y", "Metrology (j=2, m2) [Metro Frame]", K2_O),
    ("metro_z", "Z", "Metrology (j=2, m2) [Metro Frame]", K2_O),
    ("optics_x","X", "Optics box (j=2, m3) [Housing]", K3_O),
    ("optics_y","Y", "Optics box (j=2, m3) [Housing]", K3_O),
    ("optics_z","Z", "Optics box (j=2, m3) [Housing]", K3_O),
    ("lens_z",  "Z", "Lens element (j=2, m4) [Lens]", K4_O),
    
    ("w_ls_x",  "X", "Long Stroke Actuated (j=3, m2) [Wafer LS Act]", 0),
    ("w_ls_y",  "Y", "Long Stroke Actuated (j=3, m2) [Wafer LS Act]", 0),
    ("w_lsf_x", "X", "Long Stroke Flex (j=3, m3) [Wafer LS Flex]", K_LS_F),
    ("w_lsf_y", "Y", "Long Stroke Flex (j=3, m3) [Wafer LS Flex]", K_LS_F),
    
    ("w_ss_x",  "X", "Short Stroke Actuated (j=3, m4) [Wafer SS Act]", 0),
    ("w_ss_y",  "Y", "Short Stroke Actuated (j=3, m4) [Wafer SS Act]", 0),
    ("w_ssf_x", "X", "Short Stroke Flex (j=3, m5) [Wafer SS Flex]", K_SS_F),
    ("w_ssf_y", "Y", "Short Stroke Flex (j=3, m5) [Wafer SS Flex]", K_SS_F),
    
    ("w_st_x",  "X", "Fine Stage Actuated (j=3, m6) [Wafer Fine Act]", 0),
    ("w_st_y",  "Y", "Fine Stage Actuated (j=3, m6) [Wafer Fine Act]", 0),
    ("w_stf_x", "X", "Fine Stage Flex (j=3, m7) [Chuck Flex]", K_ST_F),
    ("w_stf_y", "Y", "Fine Stage Flex (j=3, m7) [Chuck Flex]", K_ST_F),
    
    ("wafer_x", "X", "Wafer (j=3, m8) [Substrate]", K8_W),
    ("wafer_y", "Y", "Wafer (j=3, m8) [Substrate]", K8_W),
    
    ("r_ls_x",  "X", "Long Stroke Actuated (j=1, m2) [Reticle LS Act]", 0),
    ("r_ls_y",  "Y", "Long Stroke Actuated (j=1, m2) [Reticle LS Act]", 0),
    ("r_lsf_x", "X", "Long Stroke Flex (j=1, m3) [Reticle LS Flex]", K_LS_F),
    ("r_lsf_y", "Y", "Long Stroke Flex (j=1, m3) [Reticle LS Flex]", K_LS_F),
    
    ("r_ss_x",  "X", "Short Stroke Actuated (j=1, m4) [Reticle SS Act]", 0),
    ("r_ss_y",  "Y", "Short Stroke Actuated (j=1, m4) [Reticle SS Act]", 0),
    ("r_ssf_x", "X", "Short Stroke Flex (j=1, m5) [Reticle SS Flex]", K_SS_F),
    ("r_ssf_y", "Y", "Short Stroke Flex (j=1, m5) [Reticle SS Flex]", K_SS_F),
    
    ("r_st_x",  "X", "Fine Stage Actuated (j=1, m6) [Reticle Fine Act]", 0),
    ("r_st_y",  "Y", "Fine Stage Actuated (j=1, m6) [Reticle Fine Act]", 0),
    ("r_stf_x", "X", "Fine Stage Flex (j=1, m7) [Mask Stage Flex]", K_ST_F),
    ("r_stf_y", "Y", "Fine Stage Flex (j=1, m7) [Mask Stage Flex]", K_ST_F),
    
    ("mask_x",  "X", "Mask (j=1, m8) [Reticle]", K8_R),
    ("mask_y",  "Y", "Mask (j=1, m8) [Reticle]", K8_R),
]

def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data  = mujoco.MjData(model)
    
    dies = get_wafer_dies(WAFER_R, DIE_L)
    controller = StepperController(dies)
    
    print(f"Total dies to expose: {len(dies)}")
    step_counter = 0

    sys.stdout.write("\033[2J\033[H")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Get reference for the Wafer Stage center
            xw, yw = controller.get_ref(data.time)
            
            # Synchronized reticle reference (4:1 anti-phase)
            xr = -xw / ALPHA
            yr = -yw / ALPHA
            
            # Set actuators
            # Wafer chain: LS gets full target, SS and Fine Stage get 0 (hold position relative to parent)
            data.ctrl[0] = xw; data.ctrl[1] = yw   # w_ls
            data.ctrl[2] = 0;  data.ctrl[3] = 0    # w_ss
            data.ctrl[4] = 0;  data.ctrl[5] = 0    # w_st
            
            # Reticle chain:
            data.ctrl[6] = xr; data.ctrl[7] = yr   # r_ls
            data.ctrl[8] = 0;  data.ctrl[9] = 0    # r_ss
            data.ctrl[10] = 0; data.ctrl[11] = 0   # r_st
            
            mujoco.mj_step(model, data)
            step_counter += 1
            
            if step_counter % 200 == 0:
                SEP = "-" * 110
                HDR = f"{'Stage':<55} {'Ax':<2} {'disp (mm)':>12} {'K (N/m)':>12} {'F=K*f (N)':>14}"
                lines = [
                    f"TIME: {data.time:7.3f}s | DIE: {controller.die_idx}/{len(dies)} | STATE: {controller.state}",
                    HDR, SEP
                ]
                
                for j_name, ax, s_name, k_val in MONITOR_CONFIG:
                    q_idx = model.joint(j_name).qposadr[0]
                    disp_m = data.qpos[q_idx]
                    force = k_val * disp_m
                    label = s_name if ax == "X" or "Lens element" in s_name or "Base frame" in s_name else ""
                    lines.append(f"{label:<55} {ax:<2} {disp_m*1000:>12.6f} {k_val:>12.2e} {force:>14.4f}")
                
                sys.stdout.write("\033[H" + "\\n".join(lines) + "\\n")
                sys.stdout.flush()

            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
"""

with open('litho_sim.py', 'w') as f:
    f.write(content)

