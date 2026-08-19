"""Control logic for the exact Al-Rawashdeh State-Space Plant.

Implements Case 1, Case 2 (FF), Case 3a (SS PI), and Case 3b (SS HIGS) on top of
rawashdeh_ss_sim.StateSpaceLithoPlant.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profiles
import rawashdeh_ss_sim as litho_sim

ALPHA = 0.25
DT = 1e-4
HOLD_TIME = 0.4
KP_ACT = 1e7


class HIGS:
    def __init__(self, dt, kp, wi, wh, kh, clip):
        self.dt = dt
        self.kp = kp
        self.wi = wi
        self.wh = wh
        self.kh = kh
        self.clip = clip
        self.x1 = 0.0
        self.x2 = 0.0

    def update(self, e):
        if e * self.x1 >= (self.x1 ** 2) / self.kh:
            self.x1 += self.wh * e * self.dt
        else:
            self.x1 = self.kh * e
        dx2 = self.wi * self.x1 * self.dt
        u = self.kp * e + self.x2 + dx2
        if abs(u) < self.clip or u * dx2 < 0:
            self.x2 += dx2
        return float(np.clip(self.kp * e + self.x2, -self.clip, self.clip))


class PI:
    def __init__(self, dt, kp, ki, clip):
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.clip = clip
        self.i = 0.0

    def update(self, e):
        di = self.ki * e * self.dt
        u = self.kp * e + self.i + di
        if abs(u) < self.clip or u * di < 0:
            self.i += di
        return float(np.clip(self.kp * e + self.i, -self.clip, self.clip))


FINE_PI = dict(kp=16.0, ki=480.0)
FINE_HIGS = dict(kp=16.0, wi=2 * np.pi * 200.0, wh=2 * np.pi * 80.0, kh=1.0)
FINE_CLIP_W = 5e-3
FINE_CLIP_R = 2e-2


def ref_kin(controller, t):
    xw, yw, xr, yr = controller.get_ref(t)
    dt_loc = t - controller.t_start
    dir_y = 1 if (controller.die_idx % 2 == 0) else -1
    vw = aw = vr = ar = vxw = 0.0
    if controller.state == "SCANNING":
        _, v, a, _, _ = controller.profile.get_kinematics(dt_loc)
        vw, aw = dir_y * v, dir_y * a
        vr, ar = -dir_y * v / ALPHA, -dir_y * a / ALPHA
    elif controller.state == "STEPPING":
        cx, cy = controller.dies[controller.die_idx]
        next_idx = (controller.die_idx + 1) % len(controller.dies)
        nx, ny = controller.dies[next_idx]
        next_dir_y = 1 if (next_idx % 2 == 0) else -1
        start_yw = -cy + dir_y * (-controller.d_offset + controller.profile.distance)
        end_yw = -ny - next_dir_y * controller.d_offset
        start_yr = -dir_y * (-controller.d_offset + controller.profile.distance) / ALPHA
        end_yr = next_dir_y * controller.d_offset / ALPHA
        T = controller.t_step
        prog = min(1.0, dt_loc / T)
        dsm = (np.pi / (2 * T)) * np.sin(np.pi * prog)
        d2sm = (np.pi ** 2 / (2 * T ** 2)) * np.cos(np.pi * prog)
        vw, aw = (end_yw - start_yw) * dsm, (end_yw - start_yw) * d2sm
        vr, ar = (end_yr - start_yr) * dsm, (end_yr - start_yr) * d2sm
        vxw = (-nx - (-cx)) * dsm
    return xw, yw, xr, yr, vw, aw, vr, ar, vxw


class StepperControllerSS:
    def __init__(self, dies, die_l, v_max, a_max, j_max, s_max, alpha=0.25, t_step=0.15):
        self.dies = dies
        self.die_l = die_l
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.s_max = s_max
        self.alpha = alpha
        self.t_step = t_step

        self.d_offset = 0.005
        scan_dist = die_l + 2 * self.d_offset
        self.profile = profiles.Profile4(v_max, a_max, j_max, s_max, scan_dist)
        self.die_idx = 0
        self.state = "HOLD"
        self.t_start = 0.0

    def get_ref(self, t):
        if self.state == "HOLD":
            cx, cy = self.dies[0]
            yw0 = -cy - self.d_offset
            yr0 = self.d_offset / self.alpha
            return cx, yw0, 0.0, yr0

        if self.state == "DONE":
            last_idx = len(self.dies) - 1
            cx, cy = self.dies[last_idx]
            dir_y = 1 if (last_idx % 2 == 0) else -1
            yw_end = -cy + dir_y * (-self.d_offset + self.profile.distance)
            yr_end = -dir_y * (-self.d_offset + self.profile.distance) / self.alpha
            return cx, yw_end, 0.0, yr_end

        dt_loc = t - self.t_start
        cx, cy = self.dies[self.die_idx]
        dir_y = 1 if (self.die_idx % 2 == 0) else -1

        if self.state == "SCANNING":
            if dt_loc >= self.profile.t_total:
                self.state = "STEPPING"
                self.t_start = t
                return self.get_ref(t)
            s, _, _, _, _ = self.profile.get_kinematics(dt_loc)
            yw = -cy + dir_y * (-self.d_offset + s)
            yr = -dir_y * (-self.d_offset + s) / self.alpha
            return cx, yw, 0.0, yr

        elif self.state == "STEPPING":
            if dt_loc >= self.t_step:
                self.die_idx += 1
                if self.die_idx >= len(self.dies):
                    self.state = "DONE"
                    return self.get_ref(t)
                self.state = "SCANNING"
                self.t_start = t
                return self.get_ref(t)

            next_idx = (self.die_idx + 1) % len(self.dies)
            nx, ny = self.dies[next_idx]
            next_dir_y = 1 if (next_idx % 2 == 0) else -1
            start_yw = -cy + dir_y * (-self.d_offset + self.profile.distance)
            end_yw = -ny - next_dir_y * self.d_offset
            start_yr = -dir_y * (-self.d_offset + self.profile.distance) / self.alpha
            end_yr = next_dir_y * self.d_offset / self.alpha

            prog = min(1.0, dt_loc / self.t_step)
            sm = 0.5 * (1 - np.cos(np.pi * prog))
            xw = cx + (nx - cx) * sm
            yw = start_yw + (end_yw - start_yw) * sm
            yr = start_yr + (end_yr - start_yr) * sm
            return xw, yw, 0.0, yr


def calibrate_lag(v=0.8, a=20.0, j=1600.0, s=1e5, iterations=3, verbose=False):
    die_l = 0.032
    dies = [(-0.016, 0.0), (0.016, 0.0)]
    ff_w = np.zeros(2)
    ff_r = np.zeros(2)
    for it in range(iterations):
        cfg = "case1" if it == 0 else "case2"
        _, extras = _run(dies, die_l, v, a, j, s, cfg,
                         ff_w=tuple(ff_w), ff_r=tuple(ff_r),
                         collect="all", want_lag_samples=True)
        (Vw, Aw, Lw), (Vr, Ar, Lr) = extras
        cw, *_ = np.linalg.lstsq(np.column_stack([Vw, Aw]), Lw, rcond=None)
        cr, *_ = np.linalg.lstsq(np.column_stack([Vr, Ar]), Lr, rcond=None)
        ff_w += cw
        ff_r += cr
        if verbose:
            print(f"  calib pass {it+1}: residual rms "
                  f"w={np.sqrt(np.mean(Lw**2))*1e6:.2f} um  "
                  f"r={np.sqrt(np.mean(Lr**2))*1e6:.2f} um", flush=True)
    return (float(ff_w[0]), float(ff_w[1])), (float(ff_r[0]), float(ff_r[1]))


def run_wafer(dies, die_l, v, a, j, s, config="case1", ff_w=(0.0, 0.0), ff_r=(0.0, 0.0),
              collect="all", perturb=None, rng=None, t_step=None,
              profile_factory=None):
    per, _ = _run(dies, die_l, v, a, j, s, config, ff_w=ff_w, ff_r=ff_r,
                  collect=collect, perturb=perturb, rng=rng, t_step=t_step,
                  profile_factory=profile_factory)
    return per


def _run(dies, die_l, v, a, j, s, config, ff_w=(0.0, 0.0), ff_r=(0.0, 0.0),
         collect="all", perturb=None, rng=None, t_step=None,
         want_lag_samples=False, profile_factory=None):
    plant = litho_sim.StateSpaceLithoPlant()
    controller = StepperControllerSS(dies, die_l, v, a, j, s, ALPHA, t_step or 0.15)
    if profile_factory is not None:
        controller.profile = profile_factory(v, a, j, s, controller.profile.distance)

    fine_w = fine_r = None
    if config == "case3a":
        fine_w = PI(DT, clip=FINE_CLIP_W, **FINE_PI)
        fine_r = PI(DT, clip=FINE_CLIP_R, **FINE_PI)
    elif config == "case3b":
        fine_w = HIGS(DT, clip=FINE_CLIP_W, **FINE_HIGS)
        fine_r = HIGS(DT, clip=FINE_CLIP_R, **FINE_HIGS)

    x = np.zeros(96)
    u = np.zeros(6)

    # Initial hold calibration
    hold_wy, hold_ry = [], []
    t = 0.0
    controller.state = "HOLD"
    while t < HOLD_TIME:
        xw0, yw0, xr0, yr0 = controller.get_ref(t)
        u[0] = KP_ACT * (yw0 - x[plant.IDX_WAFR_Y])
        u[3] = KP_ACT * (yr0 - x[plant.IDX_MASK_Y])
        x = plant.step(x, u, DT)
        t += DT
        if t > HOLD_TIME - 0.05:
            hold_wy.append(x[plant.IDX_WAFR_Y] - yw0)
            hold_ry.append(x[plant.IDX_MASK_Y] - yr0)

    off_wy = float(np.mean(hold_wy))
    off_ry = float(np.mean(hold_ry))

    # Continuous run across dies
    controller.t_start = t
    controller.state = "SCANNING"
    controller.die_idx = 0

    per_die, cur = [], []
    lag_samples = ([], [], []), ([], [], [])
    last_die_idx = -1
    t_end = t + len(dies) * (controller.profile.t_total + controller.t_step + 0.1)

    while t < t_end and controller.state != "DONE":
        xw, yw, xr, yr, vw, aw, vr, ar, vxw = ref_kin(controller, t)

        yw_cmd, yr_cmd = yw, yr
        if config in ("case2", "case3a", "case3b"):
            yw_cmd = yw - (ff_w[0] * vw + ff_w[1] * aw)
            yr_cmd = yr - (ff_r[0] * vr + ff_r[1] * ar)

        u[0] = KP_ACT * (yw_cmd - x[plant.IDX_WAFR_Y])
        u[3] = KP_ACT * (yr_cmd - x[plant.IDX_MASK_Y])

        e_w = (x[plant.IDX_WAFR_Y] - off_wy) - yw
        e_r = (x[plant.IDX_MASK_Y] - off_ry) - yr

        if fine_w is not None:
            u[1] = fine_w.update(-e_w)
            u[4] = fine_r.update(-e_r)

        if want_lag_samples:
            lag_samples[0][0].append(vw); lag_samples[0][1].append(aw)
            lag_samples[0][2].append(e_w)
            lag_samples[1][0].append(vr); lag_samples[1][1].append(ar)
            lag_samples[1][2].append(e_r)

        if controller.state == "SCANNING":
            cur.append(e_w - ALPHA * e_r)

        x = plant.step(x, u, DT)
        t += DT

        if controller.die_idx != last_die_idx and last_die_idx != -1:
            seg = np.array(cur)
            if collect == "cruise" and len(seg) >= 3:
                seg = seg[len(seg) // 3: 2 * len(seg) // 3]
            per_die.append(seg)
            cur = []
            if len(per_die) >= len(dies):
                break
        last_die_idx = controller.die_idx

    extras = None
    if want_lag_samples:
        extras = (tuple(np.array(arr) for arr in lag_samples[0]),
                  tuple(np.array(arr) for arr in lag_samples[1]))
    return per_die, extras


def moving_stats(seg, window):
    seg = np.asarray(seg)
    if len(seg) < window or window < 2:
        return float(np.abs(seg.mean())), float(seg.std())
    ma = np.convolve(seg, np.ones(window) / window, mode="valid")
    m2 = np.convolve(seg**2, np.ones(window) / window, mode="valid")
    msd = np.sqrt(np.maximum(m2 - ma**2, 0.0))
    return float(np.max(np.abs(ma))), float(np.max(msd))
