"""Al-Rawashdeh et al. (2022) Exact State-Space Plant Model.

Re-creates the 16-body planar multi-body system equations:
  M q'' + C q' + K q = B u
State vector x = [q, q'] in R^96 (16 bodies * 3 DOFs * 2).

Parameters match Appendix B of Al-Rawashdeh et al. (2022) exactly, including
the rigid actuator mount stiffness K_act = 1e12 N/m and C_act = 1e10 N*s/m.

Uses exact discrete-time state transition matrices A_d and B_d pre-computed via
block matrix exponential (scaling and squaring) to guarantee 100% numerical
stability at any integration timestep.
"""
import numpy as np

# ============================================================
# PAPER PARAMETERS - Appendix B (Al-Rawashdeh et al. 2022)
# ============================================================
M_BASE = 1400.0
M_ACT  = 0.5
M_LS_F = 49.0
M_SS_F = 28.0
M_ST_F = 10.5
M_MASK = 0.04
M_WAFR = 0.2

M_METRO = 520.0
M_OPT_B = 200.0
M_LENS  = 0.3

# Inertia tensors (z-axis scalar for planar motion)
I_BASE_Z = 3093.3
I_ACT_ZZ  = 0.00801666
I_LS_F_ZZ = 8.20668
I_SS_F_ZZ = 4.68954
I_ST_F_ZZ = 1.758574
I_MASK_ZZ = 0.000267334
I_WAFR_ZZ = 0.01024332

I_METRO_ZZ = 466.267
I_OPT_B_ZZ = 16.0
I_LENS_ZZ  = 0.0108

# Stiffnesses (N/m, Nm/rad) and Dampings (N s/m, N m s/rad)
K1 = 0.00553 * 1e9;     C1 = 0.001244 * 1e8
K1_ROT = 1e9;           C1_ROT = 1e8

# Actuated stage mounts (Al-Rawashdeh Appendix B: K = 1e12 N/m, C = 1e10 N s/m)
K_ACT_TRANS = 1e12;     C_ACT_TRANS = 1e10
K_ACT_ROT   = 1e9;      C_ACT_ROT   = 1e8

# Flex stages
K_LS_F = 0.01398 * 1e9; C_LS_F = 0.0003701 * 1e8
K_SS_F = 0.01462 * 1e9; C_SS_F = 0.0002861 * 1e8
K_ST_F = 0.01343 * 1e9; C_ST_F = 0.0001679 * 1e8

# Optics chain
K2_O = 0.4619 * 1e9;    C2_O = 0.0069308 * 1e8
K3_O = 6.396e7;         C3_O = 1.5994e5
K4_O = 1.07e6;          C4_O = 800.0

# End effectors
K8_W = 0.03158 * 1e9;   C8_W = 0.0000355 * 1e8
K8_R = 0.0063165 * 1e9; C8_R = 710.0

K_ROT_GEN = 1e9;        C_ROT_GEN = 1e8

WAFER_R = 0.150
DIE_L = 0.032
ALPHA = 0.25

def get_wafer_dies(radius=WAFER_R, die_l=DIE_L):
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


def expm_scale_square(A):
    norm_A = np.linalg.norm(A, np.inf)
    m = max(0, int(np.ceil(np.log2(norm_A / 0.5))))
    A_scaled = A / (2**m)
    res = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for i in range(1, 14):
        term = term @ A_scaled / i
        res += term
    for _ in range(m):
        res = res @ res
    return res


class StateSpaceLithoPlant:
    """16-body planar state-space plant model matching Al-Rawashdeh et al. (2022)."""

    def __init__(self, dt=1e-4):
        self.dt = dt
        self.n_bodies = 16
        self.n_dof = 48
        self.n_states = 96

        self.M = np.zeros((self.n_dof, self.n_dof))
        self.K = np.zeros((self.n_dof, self.n_dof))
        self.C = np.zeros((self.n_dof, self.n_dof))
        self.B = np.zeros((self.n_dof, 6))

        self._build_matrices()
        self._build_state_space()

    def _build_matrices(self):
        bodies_def = [
            (0, M_BASE, I_BASE_Z),          # Base
            # Reticle chain
            (1, M_ACT, I_ACT_ZZ),           # r_ls_act
            (2, M_LS_F, I_LS_F_ZZ),         # r_ls_flex
            (3, M_ACT, I_ACT_ZZ),           # r_ss_act
            (4, M_SS_F, I_SS_F_ZZ),         # r_ss_flex
            (5, M_ST_F, I_ST_F_ZZ),         # r_stage
            (6, M_MASK, I_MASK_ZZ),         # mask
            # Optics chain
            (7, M_METRO, I_METRO_ZZ),       # metro
            (8, M_OPT_B, I_OPT_B_ZZ),       # optics box
            (9, M_LENS, I_LENS_ZZ),         # lens
            # Wafer chain
            (10, M_ACT, I_ACT_ZZ),          # w_ls_act
            (11, M_LS_F, I_LS_F_ZZ),        # w_ls_flex
            (12, M_ACT, I_ACT_ZZ),          # w_ss_act
            (13, M_SS_F, I_SS_F_ZZ),        # w_ss_flex
            (14, M_ST_F, I_ST_F_ZZ),        # w_stage
            (15, M_WAFR, I_WAFR_ZZ),        # wafer
        ]

        for i, m, iz in bodies_def:
            self.M[3*i, 3*i] = m
            self.M[3*i+1, 3*i+1] = m
            self.M[3*i+2, 3*i+2] = iz

        def add_coupling(b1, b2, k_tr, c_tr, k_rot, c_rot):
            K_elem = np.diag([k_tr, k_tr, k_rot])
            C_elem = np.diag([c_tr, c_tr, c_rot])

            if b1 is not None:
                self.K[3*b1:3*b1+3, 3*b1:3*b1+3] += K_elem
                self.C[3*b1:3*b1+3, 3*b1:3*b1+3] += C_elem
            if b2 is not None:
                self.K[3*b2:3*b2+3, 3*b2:3*b2+3] += K_elem
                self.C[3*b2:3*b2+3, 3*b2:3*b2+3] += C_elem
            if b1 is not None and b2 is not None:
                self.K[3*b1:3*b1+3, 3*b2:3*b2+3] -= K_elem
                self.K[3*b2:3*b2+3, 3*b1:3*b1+3] -= K_elem
                self.C[3*b1:3*b1+3, 3*b2:3*b2+3] -= C_elem
                self.C[3*b2:3*b2+3, 3*b1:3*b1+3] -= C_elem

        # Ground -> Base
        add_coupling(None, 0, K1, C1, K1_ROT, C1_ROT)

        # Reticle chain connections:
        add_coupling(0, 1, K_ACT_TRANS, C_ACT_TRANS, K_ACT_ROT, C_ACT_ROT)
        add_coupling(1, 2, K_LS_F, C_LS_F, K_ROT_GEN, C_ROT_GEN)
        add_coupling(2, 3, K_ACT_TRANS, C_ACT_TRANS, K_ACT_ROT, C_ACT_ROT)
        add_coupling(3, 4, K_SS_F, C_SS_F, K_ROT_GEN, C_ROT_GEN)
        add_coupling(4, 5, K_ST_F, C_ST_F, K_ROT_GEN, C_ROT_GEN)
        add_coupling(5, 6, K8_R, C8_R, K_ROT_GEN, C_ROT_GEN)

        # Optics chain connections:
        add_coupling(0, 7, K2_O, C2_O, K_ROT_GEN, C_ROT_GEN)
        add_coupling(7, 8, K3_O, C3_O, 20.0, 60.0)
        add_coupling(8, 9, K4_O, C4_O, 20.0, 60.0)

        # Wafer chain connections:
        add_coupling(0, 10, K_ACT_TRANS, C_ACT_TRANS, K_ACT_ROT, C_ACT_ROT)
        add_coupling(10, 11, K_LS_F, C_LS_F, K_ROT_GEN, C_ROT_GEN)
        add_coupling(11, 12, K_ACT_TRANS, C_ACT_TRANS, K_ACT_ROT, C_ACT_ROT)
        add_coupling(12, 13, K_SS_F, C_SS_F, K_ROT_GEN, C_ROT_GEN)
        add_coupling(13, 14, K_ST_F, C_ST_F, K_ROT_GEN, C_ROT_GEN)
        add_coupling(14, 15, K8_W, C8_W, K_ROT_GEN, C_ROT_GEN)

        # Actuator force matrix B
        self.B[3*10 + 1, 0] = 1.0   # w_ls_y
        self.B[3*12 + 1, 1] = 1.0   # w_ss_y
        self.B[3*14 + 1, 2] = 1.0   # w_stage_y
        self.B[3*1 + 1, 3] = 1.0    # r_ls_y
        self.B[3*3 + 1, 4] = 1.0    # r_ss_y
        self.B[3*5 + 1, 5] = 1.0    # r_stage_y

    def _build_state_space(self):
        M_inv = np.linalg.inv(self.M)

        self.A = np.zeros((self.n_states, self.n_states))
        self.A[:self.n_dof, self.n_dof:] = np.eye(self.n_dof)
        self.A[self.n_dof:, :self.n_dof] = -M_inv @ self.K
        self.A[self.n_dof:, self.n_dof:] = -M_inv @ self.C

        self.B_ss = np.zeros((self.n_states, 6))
        self.B_ss[self.n_dof:, :] = M_inv @ self.B

        # Pre-compute discrete transition matrices A_d, B_d
        n_x = self.n_states
        n_u = 6
        M_exp = np.zeros((n_x + n_u, n_x + n_u))
        M_exp[:n_x, :n_x] = self.A
        M_exp[:n_x, n_x:] = self.B_ss

        M_d = expm_scale_square(M_exp * self.dt)
        self.A_d = M_d[:n_x, :n_x]
        self.B_d = M_d[:n_x, n_x:]

        # State indices
        self.IDX_MASK_Y = 3*6 + 1
        self.IDX_WAFR_Y = 3*15 + 1
        self.IDX_MASK_VY = self.n_dof + self.IDX_MASK_Y
        self.IDX_WAFR_VY = self.n_dof + self.IDX_WAFR_Y

    def step(self, x, u, dt=None):
        """1-step exact discrete matrix integration."""
        return self.A_d @ x + self.B_d @ u
