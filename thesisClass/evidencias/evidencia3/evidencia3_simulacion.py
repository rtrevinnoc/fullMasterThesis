import mujoco
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

# --- Translational Stiffness (K, N/m) and Damping (C, N·s/m) ---
# Diagonal entries 1-3 of the 6×6 K and C matrices (Appendix B)
# Wafer Chain End-Effector (j=3, m8)
K8_W = 0.03158 * 1e9
C8_W = 0.0000355 * 1e8

# Reticle Chain End-Effector (j=1, m8)
K8_R = 0.0063165 * 1e9
C8_R = 710.0

# General Flex stages (j=1,3: m3, m5, m7) — diagonal entries 1-3 are equal
K_LS_F = 0.01398 * 1e9; C_LS_F = 0.0003701 * 1e8
K_SS_F = 0.01462 * 1e9; C_SS_F = 0.0002861 * 1e8
K_ST_F = 0.01343 * 1e9; C_ST_F = 0.0001679 * 1e8

# Base-Floor (K1, C1) — diagonal entries 1-3
K1 = 0.00553 * 1e9; C1 = 0.001244 * 1e8

# Optics Chain (j=2) — diagonal entries 1-3
K2_O = 0.4619 * 1e9; C2_O = 0.0069308 * 1e8
K3_O = 6.396e7;      C3_O = 1.5994e5
K4_O = 1.07e6;       C4_O = 800.0

# --- Rotational Stiffness (K, N·m/rad) and Damping (C, N·m·s/rad) ---
# Diagonal entries 4-6 of the 6×6 K and C matrices (Appendix B)
K1_ROT     = 1e9;  C1_ROT     = 1e8   # Base-floor
K_LS_F_ROT = 1e9;  C_LS_F_ROT = 1e8   # LS Flex (j=1,3)
K_SS_F_ROT = 1e9;  C_SS_F_ROT = 1e8   # SS Flex
K_ST_F_ROT = 1e9;  C_ST_F_ROT = 1e8   # Fine Flex
K8_W_ROT   = 1e9;  C8_W_ROT   = 1e8   # Wafer end-effector
K8_R_ROT   = 1e8;  C8_R_ROT   = 1e8   # Mask end-effector
K2_O_ROT   = 1e9;  C2_O_ROT   = 1e8   # Metrology
K3_O_ROT   = 20.0; C3_O_ROT   = 60.0  # Optics box (diag entries 4-6 = 20, 60)
K4_O_ROT   = 20.0; C4_O_ROT   = 60.0  # Lens

# --- Inertia tensors (local frame, Appendix B) ---
# Stage actuated bodies m2/m4/m6: i^I_i = 0.00400833 * diag([1,1,2])
I_ACT_XX  = 0.00400833;  I_ACT_ZZ  = 0.00801666
# Long Stroke Flex m3: i^I_i = 4.10334 * diag([1,1,2])
I_LS_F_XX = 4.10334;     I_LS_F_ZZ = 8.20668
# Short Stroke Flex m5: i^I_i = 2.34477 * diag([1,1,2])
I_SS_F_XX = 2.34477;     I_SS_F_ZZ = 4.68954
# Fine Stage Flex m7: i^I_i = 0.879287 * diag([1,1,2])
I_ST_F_XX = 0.879287;    I_ST_F_ZZ = 1.758574
# End-effectors
I_MASK_XX = 0.000133667; I_MASK_ZZ = 0.000267334
I_WAFR_XX = 0.00512166;  I_WAFR_ZZ = 0.01024332
# Base frame: 1^I_1 = 10^3 * [[1.7618,0,-0.799],[0,3.9213,0],[-0.799,0,3.0933]] (fullinertia: Ixx Iyy Izz Ixy Ixz Iyz)
I_BASE_FULL = "1761.8 3921.3 3093.3 0 -799 0"
# Optics chain (diaginertia)
I_METRO_DIAG = "173.767 293.367 466.267"   # 2^I_2
I_OPT_B_DIAG = "50.6667 50.6667 16.0"     # 3^I_3
I_LENS_DIAG  = "0.006 0.006 0.0108"       # 4^I_4 = 0.006 * diag([1,1,1.8])

# --- Actuated-stage position servo gain ---
# Paper: K_{2,4,6} = 10^12 (near-rigid servo). In MuJoCo, position actuators implement
# F = kp*(q_ref - q), i.e. kp is the effective spring stiffness. kp=1e7 is the highest
# value that remains numerically stable with the implicit integrator at dt=1e-4 while
# still allowing smooth motion (kp=1e10 causes solver overdamping and freezes the stages).
KP_ACT = 1e7
# Based on C/K ratios from Appendix B:
# Translational C/K = 10^10 / 10^12 = 0.01
# Rotational C/K = 10^8 / 10^9 = 0.1
KV_ACT_TRANS = KP_ACT * 0.01 # 1e5
KV_ACT_ROT   = KP_ACT * 0.1  # 1e6

# ============================================================
# LITHOGRAPHY SPECIFICATIONS
# ============================================================
DIE_AREA = 1024.0           # mm^2 (32×32 mm², paper Section 6)
DIE_L = np.sqrt(DIE_AREA) * 1e-3  # m (= 0.032 m)
WAFER_R = 0.150             # m (300 mm wafer)
ALPHA = 0.25                # Demagnification (reticle 4× larger/faster)

V_W_SCAN = 0.8              # m/s
V_R_SCAN = V_W_SCAN / ALPHA # 3.2 m/s
A_W_MAX  = 20.0             # m/s^2
A_R_MAX  = A_W_MAX / ALPHA  # 80 m/s^2

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

    rows = {}
    for x, y in dies:
        if y not in rows: rows[y] = []
        rows[y].append(x)

    sorted_dies = []
    y_coords = sorted(rows.keys())
    for i, y in enumerate(y_coords):
        x_coords = sorted(rows[y])
        if i % 2 == 1: x_coords = x_coords[::-1]
        for x in x_coords:
            sorted_dies.append((x, y))
    return sorted_dies

die_grid_xml = ""
for cx, cy in get_wafer_dies(WAFER_R, DIE_L):
    die_grid_xml += f'                                        <geom type="box" size="{DIE_L/2 - 0.0005} {DIE_L/2 - 0.0005} 0.0011" pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'

MODEL_XML = f"""
<mujoco model="LITHO_FULL_MACHINE">
    <option integrator="implicit" timestep="0.0001" gravity="0 0 0"/>

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
        <geom type="plane" size="5 5 0.1" material="floor" contype="0" conaffinity="0"/>

        <!-- S1: Base Frame (m1=1400 kg) — 6-DOF spring-damper to floor -->
        <body name="base_frame" pos="0 0 0.2">
            <inertial pos="0 0 0" mass="{M_BASE}" fullinertia="{I_BASE_FULL}"/>
            <joint name="base_x"  type="slide" axis="1 0 0" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_y"  type="slide" axis="0 1 0" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_z"  type="slide" axis="0 0 1" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_rx" type="hinge" axis="1 0 0" stiffness="{K1_ROT}" damping="{C1_ROT}"/>
            <joint name="base_ry" type="hinge" axis="0 1 0" stiffness="{K1_ROT}" damping="{C1_ROT}"/>
            <joint name="base_rz" type="hinge" axis="0 0 1" stiffness="{K1_ROT}" damping="{C1_ROT}"/>
            <geom type="box" size="1.2 1.0 0.2" material="granite" contype="0" conaffinity="0"/>

            <!-- j=3: Wafer Chain (7 bodies, 6-DOF each) -->
            <!-- 1. LS Actuated (m2) — servo-controlled, stiffness=0 -->
            <body name="w_ls_act" pos="0 0 0.25">
                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                <joint name="w_ls_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                <joint name="w_ls_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                <joint name="w_ls_z"  type="slide" axis="0 0 1" stiffness="0" damping="{KV_ACT_TRANS}"/>
                <joint name="w_ls_rx" type="hinge" axis="1 0 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                <joint name="w_ls_ry" type="hinge" axis="0 1 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                <joint name="w_ls_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" contype="0" conaffinity="0"/>
                <!-- 2. LS Flex (m3) — 6-DOF spring-damper -->
                <body name="w_ls_flex" pos="0 0 0.03">
                    <inertial pos="0 0 0" mass="{M_LS_F}" diaginertia="{I_LS_F_XX} {I_LS_F_XX} {I_LS_F_ZZ}"/>
                    <joint name="w_lsf_x"  type="slide" axis="1 0 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_y"  type="slide" axis="0 1 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_z"  type="slide" axis="0 0 1" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_rx" type="hinge" axis="1 0 0" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <joint name="w_lsf_ry" type="hinge" axis="0 1 0" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <joint name="w_lsf_rz" type="hinge" axis="0 0 1" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" contype="0" conaffinity="0"/>
                    <!-- 3. SS Actuated (m4) — servo-controlled -->
                    <body name="w_ss_act" pos="0 0 0.04">
                        <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                        <joint name="w_ss_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_z"  type="slide" axis="0 0 1" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_rx" type="hinge" axis="1 0 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <joint name="w_ss_ry" type="hinge" axis="0 1 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <joint name="w_ss_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" contype="0" conaffinity="0"/>
                        <!-- 4. SS Flex (m5) — 6-DOF spring-damper -->
                        <body name="w_ss_flex" pos="0 0 0.03">
                            <inertial pos="0 0 0" mass="{M_SS_F}" diaginertia="{I_SS_F_XX} {I_SS_F_XX} {I_SS_F_ZZ}"/>
                            <joint name="w_ssf_x"  type="slide" axis="1 0 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_y"  type="slide" axis="0 1 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_z"  type="slide" axis="0 0 1" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_rx" type="hinge" axis="1 0 0" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <joint name="w_ssf_ry" type="hinge" axis="0 1 0" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <joint name="w_ssf_rz" type="hinge" axis="0 0 1" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" contype="0" conaffinity="0"/>
                            <!-- 5. Fine Stage Actuated (m6) — servo-controlled -->
                            <body name="w_st_act" pos="0 0 0.04">
                                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                                <joint name="w_st_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="w_st_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="w_st_z"  type="slide" axis="0 0 1" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="w_st_rx" type="hinge" axis="1 0 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <joint name="w_st_ry" type="hinge" axis="0 1 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <joint name="w_st_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <geom type="cylinder" size="0.16 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                <!-- 6. Fine Stage Flex (m7) — 6-DOF spring-damper -->
                                <body name="w_st_flex" pos="0 0 0.02">
                                    <inertial pos="0 0 0" mass="{M_ST_F}" diaginertia="{I_ST_F_XX} {I_ST_F_XX} {I_ST_F_ZZ}"/>
                                    <joint name="w_stf_x"  type="slide" axis="1 0 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                    <joint name="w_stf_y"  type="slide" axis="0 1 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                    <joint name="w_stf_z"  type="slide" axis="0 0 1" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                    <joint name="w_stf_rx" type="hinge" axis="1 0 0" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                    <joint name="w_stf_ry" type="hinge" axis="0 1 0" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                    <joint name="w_stf_rz" type="hinge" axis="0 0 1" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                    <geom type="cylinder" size="0.16 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                    <!-- 7. End Effector (m8) — 6-DOF spring-damper -->
                                    <body name="wafer" pos="0 0 0.02">
                                        <inertial pos="0 0 0" mass="{M_WAFR}" diaginertia="{I_WAFR_XX} {I_WAFR_XX} {I_WAFR_ZZ}"/>
                                        <joint name="wafer_x"  type="slide" axis="1 0 0" stiffness="{K8_W}"     damping="{C8_W}"/>
                                        <joint name="wafer_y"  type="slide" axis="0 1 0" stiffness="{K8_W}"     damping="{C8_W}"/>
                                        <joint name="wafer_z"  type="slide" axis="0 0 1" stiffness="{K8_W}"     damping="{C8_W}"/>
                                        <joint name="wafer_rx" type="hinge" axis="1 0 0" stiffness="{K8_W_ROT}" damping="{C8_W_ROT}"/>
                                        <joint name="wafer_ry" type="hinge" axis="0 1 0" stiffness="{K8_W_ROT}" damping="{C8_W_ROT}"/>
                                        <joint name="wafer_rz" type="hinge" axis="0 0 1" stiffness="{K8_W_ROT}" damping="{C8_W_ROT}"/>
                                        <geom type="cylinder" size="0.15 0.001" material="silicon" mass="{M_WAFR}" contype="0" conaffinity="0"/>
{die_grid_xml}                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- j=1: Reticle Chain (7 bodies, 6-DOF each) -->
            <!-- 1. LS Actuated (m2) — servo-controlled -->
            <body name="r_ls_act" pos="0 0 1.2">
                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                <joint name="r_ls_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                <joint name="r_ls_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                <joint name="r_ls_z"  type="slide" axis="0 0 1" stiffness="0" damping="{KV_ACT_TRANS}"/>
                <joint name="r_ls_rx" type="hinge" axis="1 0 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                <joint name="r_ls_ry" type="hinge" axis="0 1 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                <joint name="r_ls_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" contype="0" conaffinity="0"/>
                <!-- 2. LS Flex (m3) — 6-DOF spring-damper -->
                <body name="r_ls_flex" pos="0 0 -0.03">
                    <inertial pos="0 0 0" mass="{M_LS_F}" diaginertia="{I_LS_F_XX} {I_LS_F_XX} {I_LS_F_ZZ}"/>
                    <joint name="r_lsf_x"  type="slide" axis="1 0 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_y"  type="slide" axis="0 1 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_z"  type="slide" axis="0 0 1" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_rx" type="hinge" axis="1 0 0" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <joint name="r_lsf_ry" type="hinge" axis="0 1 0" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <joint name="r_lsf_rz" type="hinge" axis="0 0 1" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" contype="0" conaffinity="0"/>
                    <!-- 3. SS Actuated (m4) — servo-controlled -->
                    <body name="r_ss_act" pos="0 0 -0.04">
                        <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                        <joint name="r_ss_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_z"  type="slide" axis="0 0 1" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_rx" type="hinge" axis="1 0 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <joint name="r_ss_ry" type="hinge" axis="0 1 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <joint name="r_ss_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" contype="0" conaffinity="0"/>
                        <!-- 4. SS Flex (m5) — 6-DOF spring-damper -->
                        <body name="r_ss_flex" pos="0 0 -0.03">
                            <inertial pos="0 0 0" mass="{M_SS_F}" diaginertia="{I_SS_F_XX} {I_SS_F_XX} {I_SS_F_ZZ}"/>
                            <joint name="r_ssf_x"  type="slide" axis="1 0 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_y"  type="slide" axis="0 1 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_z"  type="slide" axis="0 0 1" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_rx" type="hinge" axis="1 0 0" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <joint name="r_ssf_ry" type="hinge" axis="0 1 0" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <joint name="r_ssf_rz" type="hinge" axis="0 0 1" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" contype="0" conaffinity="0"/>
                            <!-- 5. Fine Stage Actuated (m6) — servo-controlled -->
                            <body name="r_st_act" pos="0 0 -0.04">
                                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                                <joint name="r_st_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="r_st_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="r_st_z"  type="slide" axis="0 0 1" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="r_st_rx" type="hinge" axis="1 0 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <joint name="r_st_ry" type="hinge" axis="0 1 0" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <joint name="r_st_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <geom type="box" size="0.1 0.1 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                <!-- 6. Fine Stage Flex (m7) — 6-DOF spring-damper -->
                                <body name="r_st_flex" pos="0 0 -0.02">
                                    <inertial pos="0 0 0" mass="{M_ST_F}" diaginertia="{I_ST_F_XX} {I_ST_F_XX} {I_ST_F_ZZ}"/>
                                    <joint name="r_stf_x"  type="slide" axis="1 0 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                    <joint name="r_stf_y"  type="slide" axis="0 1 0" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                    <joint name="r_stf_z"  type="slide" axis="0 0 1" stiffness="{K_ST_F}"     damping="{C_ST_F}"/>
                                    <joint name="r_stf_rx" type="hinge" axis="1 0 0" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                    <joint name="r_stf_ry" type="hinge" axis="0 1 0" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                    <joint name="r_stf_rz" type="hinge" axis="0 0 1" stiffness="{K_ST_F_ROT}" damping="{C_ST_F_ROT}"/>
                                    <geom type="box" size="0.1 0.1 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                    <!-- 7. End Effector (m8) — 6-DOF spring-damper -->
                                    <body name="mask" pos="0 0 -0.02">
                                        <inertial pos="0 0 0" mass="{M_MASK}" diaginertia="{I_MASK_XX} {I_MASK_XX} {I_MASK_ZZ}"/>
                                        <joint name="mask_x"  type="slide" axis="1 0 0" stiffness="{K8_R}"     damping="{C8_R}"/>
                                        <joint name="mask_y"  type="slide" axis="0 1 0" stiffness="{K8_R}"     damping="{C8_R}"/>
                                        <joint name="mask_z"  type="slide" axis="0 0 1" stiffness="{K8_R}"     damping="{C8_R}"/>
                                        <joint name="mask_rx" type="hinge" axis="1 0 0" stiffness="{K8_R_ROT}" damping="{C8_R_ROT}"/>
                                        <joint name="mask_ry" type="hinge" axis="0 1 0" stiffness="{K8_R_ROT}" damping="{C8_R_ROT}"/>
                                        <joint name="mask_rz" type="hinge" axis="0 0 1" stiffness="{K8_R_ROT}" damping="{C8_R_ROT}"/>
                                        <geom type="box" size="0.05 0.05 0.005" material="glass" mass="{M_MASK}" contype="0" conaffinity="0"/>
                                        <!-- Highlight the reticle exposure area (4× die size) -->
                                        <geom type="box" size="{DIE_L*2} {DIE_L*2} 0.006" pos="0 0 -0.005" rgba="1 1 0 0.2" contype="0" conaffinity="0"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- j=2: Optics Chain — floor-isolated independently from base frame -->
            <!-- 1. Metrology Frame (m2=520 kg) — 6-DOF spring-damper to base frame -->
            <body name="metro_frame" pos="0 0 0.5">
            <inertial pos="0 0 0" mass="{M_METRO}" diaginertia="{I_METRO_DIAG}"/>
            <joint name="metro_x"  type="slide" axis="1 0 0" stiffness="{K2_O}"     damping="{C2_O}"/>
            <joint name="metro_y"  type="slide" axis="0 1 0" stiffness="{K2_O}"     damping="{C2_O}"/>
            <joint name="metro_z"  type="slide" axis="0 0 1" stiffness="{K2_O}"     damping="{C2_O}"/>
            <joint name="metro_rx" type="hinge" axis="1 0 0" stiffness="{K2_O_ROT}" damping="{C2_O_ROT}"/>
            <joint name="metro_ry" type="hinge" axis="0 1 0" stiffness="{K2_O_ROT}" damping="{C2_O_ROT}"/>
            <joint name="metro_rz" type="hinge" axis="0 0 1" stiffness="{K2_O_ROT}" damping="{C2_O_ROT}"/>
            <!-- Metrology Frame Structure (H-shape legs) -->
            <geom type="box" size="0.1 0.1 0.5" pos="0.8 0.6 0"   material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
            <geom type="box" size="0.1 0.1 0.5" pos="-0.8 0.6 0"  material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
            <geom type="box" size="0.1 0.1 0.5" pos="0.8 -0.6 0"  material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
            <geom type="box" size="0.1 0.1 0.5" pos="-0.8 -0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
            <geom type="box" size="0.9 0.7 0.05" pos="0 0 0.5"    material="metrology" density="1" contype="0" conaffinity="0"/>

            <!-- 2. Optics Box (m3=200 kg) — 6-DOF -->
            <body name="optics_box" pos="0 0 0.3">
                <inertial pos="0 0 0" mass="{M_OPT_B}" diaginertia="{I_OPT_B_DIAG}"/>
                <joint name="optics_x"  type="slide" axis="1 0 0" stiffness="{K3_O}"     damping="{C3_O}"/>
                <joint name="optics_y"  type="slide" axis="0 1 0" stiffness="{K3_O}"     damping="{C3_O}"/>
                <joint name="optics_z"  type="slide" axis="0 0 1" stiffness="{K3_O}"     damping="{C3_O}"/>
                <joint name="optics_rx" type="hinge" axis="1 0 0" stiffness="{K3_O_ROT}" damping="{C3_O_ROT}"/>
                <joint name="optics_ry" type="hinge" axis="0 1 0" stiffness="{K3_O_ROT}" damping="{C3_O_ROT}"/>
                <joint name="optics_rz" type="hinge" axis="0 0 1" stiffness="{K3_O_ROT}" damping="{C3_O_ROT}"/>
                <geom type="cylinder" size="0.2 0.1" material="metrology" mass="{M_OPT_B}" contype="0" conaffinity="0"/>

                <!-- 3. Lens Element (m4=0.3 kg) — 6-DOF -->
                <body name="lens" pos="0 0 -0.4">
                    <inertial pos="0 0 0" mass="{M_LENS}" diaginertia="{I_LENS_DIAG}"/>
                    <joint name="lens_x"  type="slide" axis="1 0 0" stiffness="{K4_O}"     damping="{C4_O}"/>
                    <joint name="lens_y"  type="slide" axis="0 1 0" stiffness="{K4_O}"     damping="{C4_O}"/>
                    <joint name="lens_z"  type="slide" axis="0 0 1" stiffness="{K4_O}"     damping="{C4_O}"/>
                    <joint name="lens_rx" type="hinge" axis="1 0 0" stiffness="{K4_O_ROT}" damping="{C4_O_ROT}"/>
                    <joint name="lens_ry" type="hinge" axis="0 1 0" stiffness="{K4_O_ROT}" damping="{C4_O_ROT}"/>
                    <joint name="lens_rz" type="hinge" axis="0 0 1" stiffness="{K4_O_ROT}" damping="{C4_O_ROT}"/>
                    <geom type="cylinder" size="0.1 0.05" material="glass" mass="{M_LENS}" contype="0" conaffinity="0"/>
                    <!-- Die indicator fixed to the optics chain -->
                    <geom name="die_indicator" type="box" size="{DIE_L/2} {DIE_L/2} 0.002" pos="0 0 0.015" rgba="1 1 0 0.6" contype="0" conaffinity="0"/>
                </body>
            </body>
        </body>
        </body> <!-- Close base_frame -->

        <!-- Stationary exposure slit projected from the lens -->
        <body name="exposure_field" pos="0 0 0.72">
            <geom type="box" size="{DIE_L/2} {DIE_L/2} 0.001" rgba="0 1 1 0.3" contype="0" conaffinity="0"/>
            <geom type="box" size="{DIE_L/2} {DIE_L/2} 0.5" pos="0 0 0.25" rgba="0 1 1 0.05" contype="0" conaffinity="0"/>
        </body>
    </worldbody>

    <actuator>
        <!-- Wafer chain (j=3): 6-DOF × 3 actuated stages = 18 actuators [0-17] -->
        <!-- w_ls: ctrl[0-5] -->
        <position name="w_ls_x"  joint="w_ls_x"  kp="{KP_ACT}"/>
        <position name="w_ls_y"  joint="w_ls_y"  kp="{KP_ACT}"/>
        <position name="w_ls_z"  joint="w_ls_z"  kp="{KP_ACT}"/>
        <position name="w_ls_rx" joint="w_ls_rx" kp="{KP_ACT}"/>
        <position name="w_ls_ry" joint="w_ls_ry" kp="{KP_ACT}"/>
        <position name="w_ls_rz" joint="w_ls_rz" kp="{KP_ACT}"/>
        <!-- w_ss: ctrl[6-11] -->
        <position name="w_ss_x"  joint="w_ss_x"  kp="{KP_ACT}"/>
        <position name="w_ss_y"  joint="w_ss_y"  kp="{KP_ACT}"/>
        <position name="w_ss_z"  joint="w_ss_z"  kp="{KP_ACT}"/>
        <position name="w_ss_rx" joint="w_ss_rx" kp="{KP_ACT}"/>
        <position name="w_ss_ry" joint="w_ss_ry" kp="{KP_ACT}"/>
        <position name="w_ss_rz" joint="w_ss_rz" kp="{KP_ACT}"/>
        <!-- w_st: ctrl[12-17] -->
        <position name="w_st_x"  joint="w_st_x"  kp="{KP_ACT}"/>
        <position name="w_st_y"  joint="w_st_y"  kp="{KP_ACT}"/>
        <position name="w_st_z"  joint="w_st_z"  kp="{KP_ACT}"/>
        <position name="w_st_rx" joint="w_st_rx" kp="{KP_ACT}"/>
        <position name="w_st_ry" joint="w_st_ry" kp="{KP_ACT}"/>
        <position name="w_st_rz" joint="w_st_rz" kp="{KP_ACT}"/>

        <!-- Reticle chain (j=1): 6-DOF × 3 actuated stages = 18 actuators [18-35] -->
        <!-- r_ls: ctrl[18-23] -->
        <position name="r_ls_x"  joint="r_ls_x"  kp="{KP_ACT}"/>
        <position name="r_ls_y"  joint="r_ls_y"  kp="{KP_ACT}"/>
        <position name="r_ls_z"  joint="r_ls_z"  kp="{KP_ACT}"/>
        <position name="r_ls_rx" joint="r_ls_rx" kp="{KP_ACT}"/>
        <position name="r_ls_ry" joint="r_ls_ry" kp="{KP_ACT}"/>
        <position name="r_ls_rz" joint="r_ls_rz" kp="{KP_ACT}"/>
        <!-- r_ss: ctrl[24-29] -->
        <position name="r_ss_x"  joint="r_ss_x"  kp="{KP_ACT}"/>
        <position name="r_ss_y"  joint="r_ss_y"  kp="{KP_ACT}"/>
        <position name="r_ss_z"  joint="r_ss_z"  kp="{KP_ACT}"/>
        <position name="r_ss_rx" joint="r_ss_rx" kp="{KP_ACT}"/>
        <position name="r_ss_ry" joint="r_ss_ry" kp="{KP_ACT}"/>
        <position name="r_ss_rz" joint="r_ss_rz" kp="{KP_ACT}"/>
        <!-- r_st: ctrl[30-35] -->
        <position name="r_st_x"  joint="r_st_x"  kp="{KP_ACT}"/>
        <position name="r_st_y"  joint="r_st_y"  kp="{KP_ACT}"/>
        <position name="r_st_z"  joint="r_st_z"  kp="{KP_ACT}"/>
        <position name="r_st_rx" joint="r_st_rx" kp="{KP_ACT}"/>
        <position name="r_st_ry" joint="r_st_ry" kp="{KP_ACT}"/>
        <position name="r_st_rz" joint="r_st_rz" kp="{KP_ACT}"/>
    </actuator>
</mujoco>
"""

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

        next_idx = self.die_idx + 1
        if next_idx < len(self.dies):
            nx, ny = self.dies[next_idx]
        else:
            nx, ny = cx, cy

        if self.state == "IDLE":
            self.state = "SCANNING"
            self.t_start = t
            y_offset = -dir_y * (self.d_accel + self.d_const/2)
            yw = -cy + y_offset
            xw = -cx
            yr = -y_offset / ALPHA
            xr = 0
            return xw, yw, xr, yr

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

            y_offset = dir_y * (dy - (self.d_accel + self.d_const/2))
            yw = -cy + y_offset
            xw = -cx
            yr = -y_offset / ALPHA
            xr = 0
            return xw, yw, xr, yr

        elif self.state == "STEPPING":
            if dt > self.t_step:
                self.die_idx += 1
                self.state = "SCANNING"
                self.t_start = t
                dt = self.t_step

            progress = min(1.0, dt / self.t_step)
            smooth_p = (1 - np.cos(progress * np.pi)) / 2

            start_xw = -cx
            start_yw = -cy + dir_y * (self.d_accel + self.d_const/2)

            next_dir_y = 1 if (next_idx % 2 == 0) else -1
            end_xw = -nx
            end_yw = -ny - next_dir_y * (self.d_accel + self.d_const/2)

            xw = start_xw + (end_xw - start_xw) * smooth_p
            yw = start_yw + (end_yw - start_yw) * smooth_p

            start_yr = -(dir_y * (self.d_accel + self.d_const/2)) / ALPHA
            end_yr = -(-next_dir_y * (self.d_accel + self.d_const/2)) / ALPHA
            yr = start_yr + (end_yr - start_yr) * smooth_p
            xr = 0

            return xw, yw, xr, yr

# Joint monitor: (joint_name, axis_label, stage description, translational K for force display)
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
    ("lens_x",  "X", "Lens element (j=2, m4) [Lens]", K4_O),
    ("lens_y",  "Y", "Lens element (j=2, m4) [Lens]", K4_O),
    ("lens_z",  "Z", "Lens element (j=2, m4) [Lens]", K4_O),

    ("w_ls_x",  "X", "Long Stroke Actuated (j=3, m2) [Wafer LS Act]", 0),
    ("w_ls_y",  "Y", "Long Stroke Actuated (j=3, m2) [Wafer LS Act]", 0),
    ("w_ls_z",  "Z", "Long Stroke Actuated (j=3, m2) [Wafer LS Act]", 0),
    ("w_lsf_x", "X", "Long Stroke Flex (j=3, m3) [Wafer LS Flex]", K_LS_F),
    ("w_lsf_y", "Y", "Long Stroke Flex (j=3, m3) [Wafer LS Flex]", K_LS_F),
    ("w_lsf_z", "Z", "Long Stroke Flex (j=3, m3) [Wafer LS Flex]", K_LS_F),

    ("w_ss_x",  "X", "Short Stroke Actuated (j=3, m4) [Wafer SS Act]", 0),
    ("w_ss_y",  "Y", "Short Stroke Actuated (j=3, m4) [Wafer SS Act]", 0),
    ("w_ss_z",  "Z", "Short Stroke Actuated (j=3, m4) [Wafer SS Act]", 0),
    ("w_ssf_x", "X", "Short Stroke Flex (j=3, m5) [Wafer SS Flex]", K_SS_F),
    ("w_ssf_y", "Y", "Short Stroke Flex (j=3, m5) [Wafer SS Flex]", K_SS_F),
    ("w_ssf_z", "Z", "Short Stroke Flex (j=3, m5) [Wafer SS Flex]", K_SS_F),

    ("w_st_x",  "X", "Fine Stage Actuated (j=3, m6) [Wafer Fine Act]", 0),
    ("w_st_y",  "Y", "Fine Stage Actuated (j=3, m6) [Wafer Fine Act]", 0),
    ("w_st_z",  "Z", "Fine Stage Actuated (j=3, m6) [Wafer Fine Act]", 0),
    ("w_stf_x", "X", "Fine Stage Flex (j=3, m7) [Chuck Flex]", K_ST_F),
    ("w_stf_y", "Y", "Fine Stage Flex (j=3, m7) [Chuck Flex]", K_ST_F),
    ("w_stf_z", "Z", "Fine Stage Flex (j=3, m7) [Chuck Flex]", K_ST_F),

    ("wafer_x", "X", "Wafer (j=3, m8) [Substrate]", K8_W),
    ("wafer_y", "Y", "Wafer (j=3, m8) [Substrate]", K8_W),
    ("wafer_z", "Z", "Wafer (j=3, m8) [Substrate]", K8_W),

    ("r_ls_x",  "X", "Long Stroke Actuated (j=1, m2) [Reticle LS Act]", 0),
    ("r_ls_y",  "Y", "Long Stroke Actuated (j=1, m2) [Reticle LS Act]", 0),
    ("r_ls_z",  "Z", "Long Stroke Actuated (j=1, m2) [Reticle LS Act]", 0),
    ("r_lsf_x", "X", "Long Stroke Flex (j=1, m3) [Reticle LS Flex]", K_LS_F),
    ("r_lsf_y", "Y", "Long Stroke Flex (j=1, m3) [Reticle LS Flex]", K_LS_F),
    ("r_lsf_z", "Z", "Long Stroke Flex (j=1, m3) [Reticle LS Flex]", K_LS_F),

    ("r_ss_x",  "X", "Short Stroke Actuated (j=1, m4) [Reticle SS Act]", 0),
    ("r_ss_y",  "Y", "Short Stroke Actuated (j=1, m4) [Reticle SS Act]", 0),
    ("r_ss_z",  "Z", "Short Stroke Actuated (j=1, m4) [Reticle SS Act]", 0),
    ("r_ssf_x", "X", "Short Stroke Flex (j=1, m5) [Reticle SS Flex]", K_SS_F),
    ("r_ssf_y", "Y", "Short Stroke Flex (j=1, m5) [Reticle SS Flex]", K_SS_F),
    ("r_ssf_z", "Z", "Short Stroke Flex (j=1, m5) [Reticle SS Flex]", K_SS_F),

    ("r_st_x",  "X", "Fine Stage Actuated (j=1, m6) [Reticle Fine Act]", 0),
    ("r_st_y",  "Y", "Fine Stage Actuated (j=1, m6) [Reticle Fine Act]", 0),
    ("r_st_z",  "Z", "Fine Stage Actuated (j=1, m6) [Reticle Fine Act]", 0),
    ("r_stf_x", "X", "Fine Stage Flex (j=1, m7) [Mask Stage Flex]", K_ST_F),
    ("r_stf_y", "Y", "Fine Stage Flex (j=1, m7) [Mask Stage Flex]", K_ST_F),
    ("r_stf_z", "Z", "Fine Stage Flex (j=1, m7) [Mask Stage Flex]", K_ST_F),

    ("mask_x",  "X", "Mask (j=1, m8) [Reticle]", K8_R),
    ("mask_y",  "Y", "Mask (j=1, m8) [Reticle]", K8_R),
    ("mask_z",  "Z", "Mask (j=1, m8) [Reticle]", K8_R),
]

def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data  = mujoco.MjData(model)

    dies = get_wafer_dies(WAFER_R, DIE_L)
    controller = StepperController(dies)

    print(f"Total dies to expose: {len(dies)}")
    print(f"DOF: {model.nq}  |  Actuators: {model.nu}")
    step_counter = 0

    sys.stdout.write("\033[2J\033[H")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Get reference for wafer and reticle stages
            xw, yw, xr, yr = controller.get_ref(data.time)

            # Wafer chain: LS gets full target; SS, Fine held at 0 (relative to parent)
            # ctrl layout: [x, y, z, rx, ry, rz] per stage
            data.ctrl[0]  = xw; data.ctrl[1]  = yw; data.ctrl[2:6]   = 0  # w_ls
            data.ctrl[6]  = 0;  data.ctrl[7]  = 0;  data.ctrl[8:12]  = 0  # w_ss
            data.ctrl[12] = 0;  data.ctrl[13] = 0;  data.ctrl[14:18] = 0  # w_st

            # Reticle chain: LS gets full target; SS, Fine held at 0
            data.ctrl[18] = xr; data.ctrl[19] = yr; data.ctrl[20:24] = 0  # r_ls
            data.ctrl[24] = 0;  data.ctrl[25] = 0;  data.ctrl[26:30] = 0  # r_ss
            data.ctrl[30] = 0;  data.ctrl[31] = 0;  data.ctrl[32:36] = 0  # r_st

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
                    label = s_name if ax == "X" else ""
                    lines.append(f"{label:<55} {ax:<2} {disp_m*1000:>12.6f} {k_val:>12.2e} {force:>14.4f}")

                sys.stdout.write("\033[H" + "\n".join(lines) + "\n")
                sys.stdout.flush()

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
