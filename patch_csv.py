import glob
import os

dirs = ["/Users/rtrevinnoc/maestria/tesis/research/experiments", "/Users/rtrevinnoc/maestria/tesis/research/experiments_alrawashdeh"]

for d in dirs:
    f = os.path.join(d, "driver_sweep.py")
    if os.path.exists(f):
        with open(f, "r") as file: txt = file.read()
        target = '    fig.savefig(os.path.join(out_dir, f"sweep_map_{tag}.png"),\n                dpi=160, bbox_inches="tight")\n    plt.close(fig)'
        insert = '    with open(os.path.join(out_dir, f"sweep_{tag}.csv"), "w") as f:\n        f.write("scan_idx,x,y,msd_nm,fail\\n")\n        for k, (di, m, fl) in enumerate(zip(dies, die_msd, status == 2)):\n            f.write(f"{k},{di[0]:.3f},{di[1]:.3f},{m*1e9:.1f},{int(fl)}\\n")'
        if insert not in txt:
            with open(f, "w") as file: file.write(txt.replace(target, target + "\n" + insert))
            print(f"Patched {f}")

    f = os.path.join(d, "driver_perturb.py")
    if os.path.exists(f):
        with open(f, "r") as file: txt = file.read()
        target = '    fig.savefig(os.path.join(out_dir, f"perturb_map_{name}.png"),\n                dpi=160, bbox_inches="tight")\n    plt.close(fig)'
        insert = '    with open(os.path.join(out_dir, f"perturb_{name}.csv"), "w") as f:\n        f.write("scan_idx,x,y,ma_shift_nm,msd_nm,fail\\n")\n        for k, (di, ma, m, fl) in enumerate(zip(dies, ma_shift, msds, status == 2)):\n            f.write(f"{k},{di[0]:.3f},{di[1]:.3f},{ma*1e9:.1f},{m*1e9:.1f},{int(fl)}\\n")'
        if insert not in txt:
            with open(f, "w") as file: file.write(txt.replace(target, target + "\n" + insert))
            print(f"Patched {f}")

    f = os.path.join(d, "driver_sweep_spec.py")
    if os.path.exists(f):
        with open(f, "r") as file: txt = file.read()
        target = '    fig.savefig(os.path.join(out_dir, f"specsweep_map_s{int(s)}.png"), dpi=160, bbox_inches="tight")\n    plt.close(fig)'
        insert = '    with open(os.path.join(out_dir, f"specsweep_s{int(s)}.csv"), "w") as f:\n        f.write("scan_idx,x,y,ma_nm,msd_nm,fail\\n")\n        for k, (di, m_a, m_s, fl) in enumerate(zip(dies, ma, msd, status == 2)):\n            f.write(f"{k},{di[0]:.3f},{di[1]:.3f},{m_a*1e9:.1f},{m_s*1e9:.1f},{int(fl)}\\n")'
        if insert not in txt:
            with open(f, "w") as file: file.write(txt.replace(target, target + "\n" + insert))
            print(f"Patched {f}")

    f = os.path.join(d, "exp_controllers.py")
    if os.path.exists(f):
        with open(f, "r") as file: txt = file.read()
        target = 'fig.savefig(os.path.join(fig_dir, "controllers_esyn.png"), dpi=160,\n            bbox_inches="tight")'
        insert = 'with open(os.path.join(fig_dir, "controllers_esyn.csv"), "w") as f:\n    f.write("time_ms," + ",".join(CONFIGS) + "\\n")\n    t_ms = np.arange(len(traces[CONFIGS[0]][1])) * lc.DT * 1e3\n    for i in range(len(t_ms)):\n        row = [f"{t_ms[i]:.3f}"] + [f"{traces[cfg][1][i]*1e9:.3f}" for cfg in CONFIGS]\n        f.write(",".join(row) + "\\n")'
        if insert not in txt:
            with open(f, "w") as file: file.write(txt.replace(target, target + "\n" + insert))
            print(f"Patched {f}")

    f = os.path.join(d, "exp_order_comparison.py")
    if os.path.exists(f):
        with open(f, "r") as file: txt = file.read()
        target = 'fig.savefig(os.path.join(fig_dir, "order_comparison_esyn.png"), dpi=160,\n            bbox_inches="tight")'
        insert = 'with open(os.path.join(fig_dir, "order_comparison_esyn.csv"), "w") as f:\n    headers = [f"{p}_{c}" for c, _ in CONTROLLERS for p, _ in PROFILES]\n    f.write("time_ms," + ",".join(headers) + "\\n")\n    t_ms = np.arange(len(traces[(PROFILES[0][0], CONTROLLERS[0][0])])) * lc.DT * 1e3\n    for i in range(len(t_ms)):\n        row = [f"{t_ms[i]:.3f}"] + [f"{traces[(p, c)][i]*1e9:.3f}" for c, _ in CONTROLLERS for p, _ in PROFILES]\n        f.write(",".join(row) + "\\n")'
        if insert not in txt:
            with open(f, "w") as file: file.write(txt.replace(target, target + "\n" + insert))
            print(f"Patched {f}")
