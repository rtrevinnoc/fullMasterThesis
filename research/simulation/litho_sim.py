import mujoco
import mujoco.viewer
import time
import numpy as np
import sys
import argparse
import os

# Exact bounded-derivative generators (Profile2/3/4). The StepperController
# uses the exact 4th-order generator by default; the polynomial Profile4thOrder
# below is retained only for reference and is no longer used.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "experiments"))
import profiles  # noqa: E402

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
K8_W = 0.03158 * 1e9
C8_W = 0.0000355 * 1e8
K8_R = 0.0063165 * 1e9
C8_R = 710.0
K_LS_F = 0.01398 * 1e9; C_LS_F = 0.0003701 * 1e8
K_SS_F = 0.01462 * 1e9; C_SS_F = 0.0002861 * 1e8
K_ST_F = 0.01343 * 1e9; C_ST_F = 0.0001679 * 1e8
K1 = 0.00553 * 1e9; C1 = 0.001244 * 1e8
K2_O = 0.4619 * 1e9; C2_O = 0.0069308 * 1e8
K3_O = 6.396e7;      C3_O = 1.5994e5
K4_O = 1.07e6;       C4_O = 800.0

# --- Rotational Stiffness ---
K1_ROT     = 1e9;  C1_ROT     = 1e8
K_LS_F_ROT = 1e9;  C_LS_F_ROT = 1e8
K_SS_F_ROT = 1e9;  C_SS_F_ROT = 1e8
K_ST_F_ROT = 1e9;  C_ST_F_ROT = 1e8
K8_W_ROT   = 1e9;  C8_W_ROT   = 1e8
K8_R_ROT   = 1e8;  C8_R_ROT   = 1e8
K2_O_ROT   = 1e9;  C2_O_ROT   = 1e8
K3_O_ROT   = 20.0; C3_O_ROT   = 60.0
K4_O_ROT   = 20.0; C4_O_ROT   = 60.0

# --- Inertia tensors ---
I_ACT_XX  = 0.00400833;  I_ACT_ZZ  = 0.00801666
I_LS_F_XX = 4.10334;     I_LS_F_ZZ = 8.20668
I_SS_F_XX = 2.34477;     I_SS_F_ZZ = 4.68954
I_ST_F_XX = 0.879287;    I_ST_F_ZZ = 1.758574
I_MASK_XX = 0.000133667; I_MASK_ZZ = 0.000267334
I_WAFR_XX = 0.00512166;  I_WAFR_ZZ = 0.01024332
I_BASE_FULL = "1761.8 3921.3 3093.3 0 -799 0"
I_METRO_DIAG = "173.767 293.367 466.267"
I_OPT_B_DIAG = "50.6667 50.6667 16.0"
I_LENS_DIAG  = "0.006 0.006 0.0108"

# --- Actuator gains ---
# The long-stroke (LS) translational joints are frictionless (air-bearing
# guideways, Al-Rawashdeh 2022 Remark 2.2): their velocity feedback KV acts on
# the TRACKING-ERROR velocity via <velocity> actuators commanded with the
# reference velocity (F = kv*(v_ref - v)), not as joint damping, which would
# add a drag force kv*v_scan during cruise (a 10 mm per m/s servo lag).
# Joints that regulate a constant reference (short-stroke, stepper, all rz)
# keep plain joint damping: for a constant setpoint both forms coincide.
KP_ACT = 1e7
KV_ACT_TRANS = KP_ACT * 0.01
KV_ACT_ROT   = KP_ACT * 0.1
# NOTE: the fine-stage (stepper) servo keeps kv as plain joint damping: its
# reference is quasi-static, and raising its bandwidth (lower kv, or kv on
# the command error) destabilizes against the flexible chain modes. The
# fine-stage authority pole kp/kv ~ 100 rad/s is therefore a structural
# property of this generic replication (unlike the reference machine's
# purpose-designed 2 kHz fine stage, Al-Rawashdeh 2022 Sec. 3).

class Profile4thOrder:
    def __init__(self, v_max, a_max, j_max, s_max, distance):
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.s_max = s_max
        self.distance = distance
        self.t_j = min(np.sqrt(a_max / j_max), a_max / j_max)
        self.t_a = (v_max - j_max * self.t_j**2) / a_max
        if self.t_a < 0:
            self.t_j = (v_max / j_max)**(1/3)
            self.t_a = 0
        self.t_accel = 2*self.t_j + self.t_a
        self.d_accel = v_max * self.t_accel / 2
        self.t_const = max(0, (distance - 2*self.d_accel) / v_max)
        self.t_total = 2*self.t_accel + self.t_const

    def get_kinematics(self, t):
        if t < 0: return 0.0, 0.0, 0.0, 0.0, 0.0
        if t > self.t_total: return self.distance, 0.0, 0.0, 0.0, 0.0
        if t < self.t_accel:
            tau = t / self.t_accel
            p = self.v_max * self.t_accel * (2.5*tau**4 - 3*tau**5 + tau**6)
            v = self.v_max * (10*tau**3 - 15*tau**4 + 6*tau**5)
            a = (self.v_max / self.t_accel) * (30*tau**2 - 60*tau**3 + 30*tau**4)
            j = (self.v_max / self.t_accel**2) * (60*tau - 180*tau**2 + 120*tau**3)
            s = (self.v_max / self.t_accel**3) * (60 - 360*tau + 360*tau**2)
            return p, v, a, j, s
        elif t < self.t_accel + self.t_const:
            p_accel = self.v_max * self.t_accel * 0.5
            p = p_accel + self.v_max * (t - self.t_accel)
            return p, self.v_max, 0.0, 0.0, 0.0
        else:
            t_dec = t - (self.t_accel + self.t_const)
            tau = t_dec / self.t_accel
            v_ratio = (10*tau**3 - 15*tau**4 + 6*tau**5)
            v = self.v_max * (1 - v_ratio)
            p_const = (self.v_max * self.t_accel * 0.5) + self.v_max * self.t_const
            p = p_const + self.v_max * self.t_accel * (tau - (2.5*tau**4 - 3*tau**5 + tau**6))
            a = -(self.v_max / self.t_accel) * (30*tau**2 - 60*tau**3 + 30*tau**4)
            j = -(self.v_max / self.t_accel**2) * (60*tau - 180*tau**2 + 120*tau**3)
            s = -(self.v_max / self.t_accel**3) * (60 - 360*tau + 360*tau**2)
            return p, v, a, j, s

    def get_pos(self, t):
        p, _, _, _, _ = self.get_kinematics(t)
        return p

class StepperController:
    def __init__(self, dies, die_l, v_scan, a_scan, j_max, s_max, alpha):
        self.dies = dies
        self.die_l = die_l
        self.v_scan = v_scan
        self.a_scan = a_scan
        self.j_max = j_max
        self.s_max = s_max
        self.alpha = alpha
        self.die_idx = 0
        self.state = "IDLE"
        self.t_start = 0
        t_ramp = (v_scan/a_scan) + (a_scan/j_max)
        d_ramp = v_scan * t_ramp
        self.profile = profiles.Profile4(v_scan, a_scan, j_max, s_max, self.die_l + d_ramp)
        self.d_offset = d_ramp / 2
        self.t_step = 0.2

    def get_ref(self, t):
        cx, cy = self.dies[self.die_idx]
        dt = t - self.t_start
        dir_y = 1 if (self.die_idx % 2 == 0) else -1
        next_idx = (self.die_idx + 1) % len(self.dies)
        nx, ny = self.dies[next_idx]
        if self.state == "IDLE":
            self.state = "SCANNING"; self.t_start = t
            yw = -cy - dir_y * self.d_offset; xw = -cx
            yr = self.d_offset / self.alpha; xr = 0
            return xw, yw, xr, yr
        elif self.state == "SCANNING":
            if dt > self.profile.t_total:
                self.state = "STEPPING"; self.t_start = t; dt = 0
            dist = self.profile.get_pos(dt)
            y_offset = -self.d_offset + dist
            yw = -cy + dir_y * y_offset; xw = -cx
            yr = -dir_y * y_offset / self.alpha; xr = 0
            return xw, yw, xr, yr
        elif self.state == "STEPPING":
            if dt > self.t_step:
                self.die_idx = next_idx; self.state = "SCANNING"; self.t_start = t; dt = self.t_step
            progress = min(1.0, dt / self.t_step)
            smooth_p = (1 - np.cos(progress * np.pi)) / 2
            start_yw = -cy + dir_y * (-self.d_offset + self.profile.distance)
            start_xw = -cx
            next_dir_y = 1 if (next_idx % 2 == 0) else -1
            end_yw = -ny - next_dir_y * self.d_offset; end_xw = -nx
            xw = start_xw + (end_xw - start_xw) * smooth_p
            yw = start_yw + (end_yw - start_yw) * smooth_p
            start_yr = -dir_y * (-self.d_offset + self.profile.distance) / self.alpha
            end_yr = next_dir_y * self.d_offset / self.alpha
            yr = start_yr + (end_yr - start_yr) * smooth_p
            xr = 0
            return xw, yw, xr, yr

class HIGSController:
    """Hybrid Integrator-Gain System (HIGS) - Section 3.3"""
    def __init__(self, dt, kp, wi, wh, kh):
        self.dt = dt
        self.kp = kp; self.wi = wi; self.wh = wh; self.kh = kh
        self.x1 = 0.0 # HIGS element state
        self.x2 = 0.0 # Linear integrator state
        self.k_lead = self.kh / (self.wh * np.sqrt(1 + 16/(np.pi**2)))

    def update(self, e, edot):
        e_bar = e + self.k_lead * edot
        u_bar = self.x1
        if e_bar * u_bar >= (1.0/self.kh) * (u_bar**2):
            self.x1 += self.wh * e_bar * self.dt
        else:
            self.x1 = self.kh * e_bar
        self.x2 += self.wi * self.x1 * self.dt
        return self.kp * e + self.x2

# --- Ideal balance masses -----------------------------------------------
# In Al-Rawashdeh 2022 the actuator forces enter the chain dynamics as
# EXTERNAL wrenches (eq. 2.5): the machine frame does not feel the stage
# acceleration reaction. Physically this is the balance mass of each chain
# (their Fig. 1). MuJoCo joint actuators react on the parent body, so we
# emulate the balance mass by cancelling the long-stroke actuator reaction
# on the base frame each step (the LS joint is the only mechanical path
# between the base and a chain: it is frictionless with zero stiffness).
BAL_JOINTS = (("w_ls_x", 0), ("w_ls_y", 1), ("r_ls_x", 0), ("r_ls_y", 1))


def make_reaction_canceller(model):
    """Return cancel(data): overwrite the base-frame applied force with the
    sum of the long-stroke actuator forces (position + velocity terms), so
    the net actuator reaction on the base is zero (ideal balance mass)."""
    base_id = model.body("base_frame").id
    entries = []
    for name, axis in BAL_JOINTS:
        joint = model.joint(name)
        entries.append((joint.qposadr[0], joint.dofadr[0],
                        model.actuator(name).id,
                        model.actuator(name + "_v").id, axis))

    def cancel(data):
        f_xy = [0.0, 0.0]
        for qadr, dadr, a_pos, a_vel, axis in entries:
            f = (KP_ACT * (data.ctrl[a_pos] - data.qpos[qadr])
                 + KV_ACT_TRANS * (data.ctrl[a_vel] - data.qvel[dadr]))
            f_xy[axis] += f
        data.xfrc_applied[base_id][0] = f_xy[0]
        data.xfrc_applied[base_id][1] = f_xy[1]

    return cancel


MONITOR_CONFIG = [
    ("base_x",  "X", "Base frame (Body 1) [Floor]", K1),
    ("base_y",  "Y", "Base frame (Body 1) [Floor]", K1),
    ("metro_x", "X", "Metrology (j=2, Body 2) [Metro Frame]", K2_O),
    ("metro_y", "Y", "Metrology (j=2, Body 2) [Metro Frame]", K2_O),
    ("optics_x","X", "Optics box (j=2, Body 3) [Housing]", K3_O),
    ("optics_y","Y", "Optics box (j=2, Body 3) [Housing]", K3_O),
    ("lens_x",  "X", "Lens element (j=2, Body 4) [Lens]", K4_O),
    ("lens_y",  "Y", "Lens element (j=2, Body 4) [Lens]", K4_O),
    ("wafer_x", "X", "Wafer die (j=3, Body 7) [end-effector]", K8_W),
    ("wafer_y", "Y", "Wafer die (j=3, Body 7) [end-effector]", K8_W),
]

def get_wafer_dies(radius, die_l):
    dies = []
    n = int(np.ceil(2 * radius / die_l)) + 2
    for i in range(-n, n):
        for j in range(-n, n):
            cx = i * die_l; cy = j * die_l
            corners = [(cx-die_l/2, cy-die_l/2), (cx+die_l/2, cy-die_l/2),
                       (cx-die_l/2, cy+die_l/2), (cx+die_l/2, cy+die_l/2)]
            if all(np.sqrt(x**2 + y**2) < radius for x, y in corners): dies.append((cx, cy))
    rows = {}
    for x, y in dies:
        if y not in rows: rows[y] = []
        rows[y].append(x)
    sorted_dies = []
    y_coords = sorted(rows.keys())
    for i, y in enumerate(y_coords):
        x_coords = sorted(rows[y])
        if i % 2 == 1: x_coords = x_coords[::-1]
        for x in x_coords: sorted_dies.append((x, y))
    return sorted_dies

def get_model_xml(die_l, die_grid_xml):
    return f"""
<mujoco model="LITHO_FULL_MACHINE">
    <option integrator="implicit" timestep="0.0001" gravity="0 0 0"/>
    <visual>
        <headlight ambient="0.3 0.3 0.3" diffuse="0.8 0.8 0.8"/>
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
        <body name="base_frame" pos="0 0 0.2">
            <inertial pos="0 0 0" mass="{M_BASE}" fullinertia="{I_BASE_FULL}"/>
            <joint name="base_x"  type="slide" axis="1 0 0" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_y"  type="slide" axis="0 1 0" stiffness="{K1}"     damping="{C1}"/>
            <joint name="base_rz" type="hinge" axis="0 0 1" stiffness="{K1_ROT}" damping="{C1_ROT}"/>
            <geom type="box" size="1.2 1.0 0.2" material="granite" contype="0" conaffinity="0"/>
            <body name="w_ls_act" pos="0 0 0.25">
                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                <joint name="w_ls_x"  type="slide" axis="1 0 0" stiffness="0" damping="0"/>
                <joint name="w_ls_y"  type="slide" axis="0 1 0" stiffness="0" damping="0"/>
                <joint name="w_ls_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" contype="0" conaffinity="0"/>
                <body name="w_ls_flex" pos="0 0 0.03">
                    <inertial pos="0 0 0" mass="{M_LS_F}" diaginertia="{I_LS_F_XX} {I_LS_F_XX} {I_LS_F_ZZ}"/>
                    <joint name="w_lsf_x"  type="slide" axis="1 0 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_y"  type="slide" axis="0 1 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="w_lsf_rz" type="hinge" axis="0 0 1" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" contype="0" conaffinity="0"/>
                    <body name="w_ss_act" pos="0 0 0.04">
                        <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                        <joint name="w_ss_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="w_ss_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" contype="0" conaffinity="0"/>
                        <body name="w_ss_flex" pos="0 0 0.03">
                            <inertial pos="0 0 0" mass="{M_SS_F}" diaginertia="{I_SS_F_XX} {I_SS_F_XX} {I_SS_F_ZZ}"/>
                            <joint name="w_ssf_x"  type="slide" axis="1 0 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_y"  type="slide" axis="0 1 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="w_ssf_rz" type="hinge" axis="0 0 1" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" contype="0" conaffinity="0"/>
                            <body name="w_st_act" pos="0 0 0.04">
                                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                                <joint name="w_st_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="w_st_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="w_st_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <geom type="cylinder" size="0.16 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                <body name="wafer" pos="0 0 0.02">
                                    <inertial pos="0 0 0" mass="{M_WAFR}" diaginertia="{I_WAFR_XX} {I_WAFR_XX} {I_WAFR_ZZ}"/>
                                    <joint name="wafer_x"  type="slide" axis="1 0 0" stiffness="{K8_W}"     damping="{C8_W}"/>
                                    <joint name="wafer_y"  type="slide" axis="0 1 0" stiffness="{K8_W}"     damping="{C8_W}"/>
                                    <joint name="wafer_rz" type="hinge" axis="0 0 1" stiffness="{K8_W_ROT}" damping="{C8_W_ROT}"/>
                                    <geom type="cylinder" size="0.15 0.001" material="silicon" contype="0" conaffinity="0"/>
{die_grid_xml}                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="r_ls_act" pos="0 0 1.2">
                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                <joint name="r_ls_x"  type="slide" axis="1 0 0" stiffness="0" damping="0"/>
                <joint name="r_ls_y"  type="slide" axis="0 1 0" stiffness="0" damping="0"/>
                <joint name="r_ls_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                <geom type="box" size="0.4 0.3 0.02" material="ls_steel" contype="0" conaffinity="0"/>
                <body name="r_ls_flex" pos="0 0 -0.03">
                    <inertial pos="0 0 0" mass="{M_LS_F}" diaginertia="{I_LS_F_XX} {I_LS_F_XX} {I_LS_F_ZZ}"/>
                    <joint name="r_lsf_x"  type="slide" axis="1 0 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_y"  type="slide" axis="0 1 0" stiffness="{K_LS_F}"     damping="{C_LS_F}"/>
                    <joint name="r_lsf_rz" type="hinge" axis="0 0 1" stiffness="{K_LS_F_ROT}" damping="{C_LS_F_ROT}"/>
                    <geom type="box" size="0.38 0.28 0.03" material="ls_steel" contype="0" conaffinity="0"/>
                    <body name="r_ss_act" pos="0 0 -0.04">
                        <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                        <joint name="r_ss_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                        <joint name="r_ss_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                        <geom type="box" size="0.2 0.2 0.02" material="ss_alum" contype="0" conaffinity="0"/>
                        <body name="r_ss_flex" pos="0 0 -0.03">
                            <inertial pos="0 0 0" mass="{M_SS_F}" diaginertia="{I_SS_F_XX} {I_SS_F_XX} {I_SS_F_ZZ}"/>
                            <joint name="r_ssf_x"  type="slide" axis="1 0 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_y"  type="slide" axis="0 1 0" stiffness="{K_SS_F}"     damping="{C_SS_F}"/>
                            <joint name="r_ssf_rz" type="hinge" axis="0 0 1" stiffness="{K_SS_F_ROT}" damping="{C_SS_F_ROT}"/>
                            <geom type="box" size="0.18 0.18 0.03" material="ss_alum" contype="0" conaffinity="0"/>
                            <body name="r_st_act" pos="0 0 -0.04">
                                <inertial pos="0 0 0" mass="{M_ACT}" diaginertia="{I_ACT_XX} {I_ACT_XX} {I_ACT_ZZ}"/>
                                <joint name="r_st_x"  type="slide" axis="1 0 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="r_st_y"  type="slide" axis="0 1 0" stiffness="0" damping="{KV_ACT_TRANS}"/>
                                <joint name="r_st_rz" type="hinge" axis="0 0 1" stiffness="0" damping="{KV_ACT_ROT}"/>
                                <geom type="box" size="0.1 0.1 0.015" material="ws_alum" contype="0" conaffinity="0"/>
                                <body name="mask" pos="0 0 -0.02">
                                    <inertial pos="0 0 0" mass="{M_MASK}" diaginertia="{I_MASK_XX} {I_MASK_XX} {I_MASK_ZZ}"/>
                                    <joint name="mask_x"  type="slide" axis="1 0 0" stiffness="{K8_R}"     damping="{C8_R}"/>
                                    <joint name="mask_y"  type="slide" axis="0 1 0" stiffness="{K8_R}"     damping="{C8_R}"/>
                                    <joint name="mask_rz" type="hinge" axis="0 0 1" stiffness="{K8_R_ROT}" damping="{C8_R_ROT}"/>
                                    <geom type="box" size="0.05 0.05 0.005" material="glass" contype="0" conaffinity="0"/>
                                    <geom type="box" size="{die_l*2} {die_l*2} 0.006" pos="0 0 -0.005" rgba="1 1 0 0.2" contype="0" conaffinity="0"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="metro_frame" pos="0 0 0.5">
                <inertial pos="0 0 0" mass="{M_METRO}" diaginertia="{I_METRO_DIAG}"/>
                <joint name="metro_x"  type="slide" axis="1 0 0" stiffness="{K2_O}"     damping="{C2_O}"/>
                <joint name="metro_y"  type="slide" axis="0 1 0" stiffness="{K2_O}"     damping="{C2_O}"/>
                <joint name="metro_rz" type="hinge" axis="0 0 1" stiffness="{K2_O_ROT}" damping="{C2_O_ROT}"/>
                <geom type="box" size="0.1 0.1 0.5" pos="0.8 0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.1 0.1 0.5" pos="-0.8 0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.1 0.1 0.5" pos="0.8 -0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.1 0.1 0.5" pos="-0.8 -0.6 0" material="metrology" mass="{M_METRO/4}" contype="0" conaffinity="0"/>
                <geom type="box" size="0.9 0.7 0.05" pos="0 0 0.5" material="metrology" density="1" contype="0" conaffinity="0"/>
                <body name="optics_box" pos="0 0 0.3">
                    <inertial pos="0 0 0" mass="{M_OPT_B}" diaginertia="{I_OPT_B_DIAG}"/>
                    <joint name="optics_x"  type="slide" axis="1 0 0" stiffness="{K3_O}"     damping="{C3_O}"/>
                    <joint name="optics_y"  type="slide" axis="0 1 0" stiffness="{K3_O}"     damping="{C3_O}"/>
                    <joint name="optics_rz" type="hinge" axis="0 0 1" stiffness="{K3_O_ROT}" damping="{C3_O_ROT}"/>
                    <geom type="cylinder" size="0.2 0.2" material="metrology" mass="{M_OPT_B}" contype="0" conaffinity="0"/>
                    <body name="lens" pos="0 0 -0.1">
                        <inertial pos="0 0 0" mass="{M_LENS}" diaginertia="{I_LENS_DIAG}"/>
                        <joint name="lens_x"  type="slide" axis="1 0 0" stiffness="{K4_O}"     damping="{C4_O}"/>
                        <joint name="lens_y"  type="slide" axis="0 1 0" stiffness="{K4_O}"     damping="{C4_O}"/>
                        <joint name="lens_rz" type="hinge" axis="0 0 1" stiffness="{K4_O_ROT}" damping="{C4_O_ROT}"/>
                        <geom type="cylinder" size="0.1 0.05" material="glass" mass="{M_LENS}" contype="0" conaffinity="0"/>
                        <geom name="die_indicator" type="box" size="{die_l/2} {die_l/2} 0.002" pos="0 0 -0.25" rgba="1 1 0 0.6" contype="0" conaffinity="0"/>
                    </body>
                </body>
            </body>
        </body>
        <body name="exposure_field" pos="0 0 0.72">
            <geom type="box" size="{die_l/2} {die_l/2} 0.5" pos="0 0 0.25" rgba="0 1 1 0.05" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
    <actuator>
        <position name="w_ls_x"  joint="w_ls_x"  kp="{KP_ACT}"/>
        <position name="w_ls_y"  joint="w_ls_y"  kp="{KP_ACT}"/>
        <position name="w_ls_rz" joint="w_ls_rz" kp="{KP_ACT}"/>
        <position name="w_ss_x"  joint="w_ss_x"  kp="{KP_ACT}"/>
        <position name="w_ss_y"  joint="w_ss_y"  kp="{KP_ACT}"/>
        <position name="w_ss_rz" joint="w_ss_rz" kp="{KP_ACT}"/>
        <position name="w_st_x"  joint="w_st_x"  kp="{KP_ACT}"/>
        <position name="w_st_y"  joint="w_st_y"  kp="{KP_ACT}"/>
        <position name="w_st_rz" joint="w_st_rz" kp="{KP_ACT}"/>
        <position name="r_ls_x"  joint="r_ls_x"  kp="{KP_ACT}"/>
        <position name="r_ls_y"  joint="r_ls_y"  kp="{KP_ACT}"/>
        <position name="r_ls_rz" joint="r_ls_rz" kp="{KP_ACT}"/>
        <position name="r_ss_x"  joint="r_ss_x"  kp="{KP_ACT}"/>
        <position name="r_ss_y"  joint="r_ss_y"  kp="{KP_ACT}"/>
        <position name="r_ss_rz" joint="r_ss_rz" kp="{KP_ACT}"/>
        <position name="r_st_x"  joint="r_st_x"  kp="{KP_ACT}"/>
        <position name="r_st_y"  joint="r_st_y"  kp="{KP_ACT}"/>
        <position name="r_st_rz" joint="r_st_rz" kp="{KP_ACT}"/>
        <velocity name="w_ls_x_v" joint="w_ls_x" kv="{KV_ACT_TRANS}"/>
        <velocity name="w_ls_y_v" joint="w_ls_y" kv="{KV_ACT_TRANS}"/>
        <velocity name="r_ls_x_v" joint="r_ls_x" kv="{KV_ACT_TRANS}"/>
        <velocity name="r_ls_y_v" joint="r_ls_y" kv="{KV_ACT_TRANS}"/>
    </actuator>
</mujoco>
"""

def main():
    parser = argparse.ArgumentParser(description="Full Lithography Machine Simulation")
    parser.add_argument("--v-scan", type=float, default=0.8, help="Wafer scan speed [m/s]")
    parser.add_argument("--a-max", type=float, default=20.0, help="Max wafer acceleration [m/s^2]")
    parser.add_argument("--j-max", type=float, default=1600.0, help="Max wafer jerk [m/s^3]")
    parser.add_argument("--s-max", type=float, default=1e5, help="Max wafer snap [m/s^4]")
    parser.add_argument("--die-area", type=float, default=1024.0, help="Die area [mm^2]")
    parser.add_argument("--control", choices=['ff', 'higs'], default='ff', help="Control strategy")
    parser.add_argument("--kp-fb", type=float, default=2e6, help="Feedback proportional gain")
    parser.add_argument("--wh", type=float, default=2*np.pi*20, help="HIGS integrator frequency [rad/s]")
    args = parser.parse_args()
    die_l = np.sqrt(args.die_area) * 1e-3
    wafer_r = 0.150; alpha = 0.25
    dies = get_wafer_dies(wafer_r, die_l)
    die_grid_xml = ""
    for cx, cy in dies:
        die_grid_xml += f'<geom type="box" size="{die_l/2 - 0.0005} {die_l/2 - 0.0005} 0.0011" pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\n'
    model_xml = get_model_xml(die_l, die_grid_xml)
    model = mujoco.MjModel.from_xml_string(model_xml)
    data  = mujoco.MjData(model)
    controller = StepperController(dies, die_l, args.v_scan, args.a_max, args.j_max, args.s_max, alpha)
    cancel_reactions = make_reaction_canceller(model)
    higs_w = HIGSController(model.opt.timestep, args.kp_fb, 2*np.pi*34.4, args.wh, 1.0)
    higs_r = HIGSController(model.opt.timestep, args.kp_fb, 2*np.pi*34.4, args.wh, 1.0)
    print(f"Total dies to expose: {len(dies)} | Mode: {args.control.upper()}")
    step_counter = 0
    prev_ref = None
    sys.stdout.write("\033[2J\033[H")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            xw, yw, xr, yr = controller.get_ref(data.time)
            data.ctrl[0] = xw; data.ctrl[1] = yw; data.ctrl[2] = 0
            data.ctrl[9] = xr; data.ctrl[10] = yr; data.ctrl[11] = 0
            # LS velocity references (error-velocity damping), finite-differenced
            if prev_ref is not None:
                dt = model.opt.timestep
                data.ctrl[18] = (xw - prev_ref[0]) / dt
                data.ctrl[19] = (yw - prev_ref[1]) / dt
                data.ctrl[20] = (xr - prev_ref[2]) / dt
                data.ctrl[21] = (yr - prev_ref[3]) / dt
            prev_ref = (xw, yw, xr, yr)
            if args.control == 'higs':
                ew_y = yw - data.body('wafer').xpos[1]
                er_y = yr - data.body('mask').xpos[1]
                ew_y_dot = -data.qvel[model.joint('wafer_y').dofadr[0]]
                er_y_dot = -data.qvel[model.joint('mask_y').dofadr[0]]
                data.ctrl[7] = higs_w.update(ew_y, ew_y_dot)
                data.ctrl[16] = higs_r.update(er_y, er_y_dot)
            else:
                data.ctrl[7] = 0; data.ctrl[16] = 0
            cancel_reactions(data)
            mujoco.mj_step(model, data)
            step_counter += 1
            if step_counter % 200 == 0:
                lines = [f"TIME: {data.time:7.3f}s | DIE: {controller.die_idx}/{len(dies)} | STATE: {controller.state}"]
                for j_name, ax, s_name, k_val in MONITOR_CONFIG:
                    disp_m = data.qpos[model.joint(j_name).qposadr[0]]
                    lines.append(f"{s_name if ax=='X' else '':<55} {ax:<2} {disp_m*1000:>12.6f}")
                sys.stdout.write("\033[H" + "\n".join(lines) + "\n"); sys.stdout.flush()
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep: time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
