import re

with open('litho_sim.py', 'r') as f:
    content = f.read()

new_config = """MONITOR_CONFIG = [
    # (joint_name, axis, stage_name, K_value)
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
    ("w_lsf_x", "X", "Long Stroke (j=3, m3) [Wafer LS]", K_LS_F),
    ("w_lsf_y", "Y", "Long Stroke (j=3, m3) [Wafer LS]", K_LS_F),
    ("w_ssf_x", "X", "Short Stroke (j=3, m5) [Wafer SS]", K_SS_F),
    ("w_ssf_y", "Y", "Short Stroke (j=3, m5) [Wafer SS]", K_SS_F),
    ("w_stf_x", "X", "Wafer Stage (j=3, m7) [Chuck]", K_ST_F),
    ("w_stf_y", "Y", "Wafer Stage (j=3, m7) [Chuck]", K_ST_F),
    ("wafer_x", "X", "Wafer (j=3, m8) [Substrate]", K8_W),
    ("wafer_y", "Y", "Wafer (j=3, m8) [Substrate]", K8_W),
    ("r_lsf_x", "X", "Long Stroke (j=1, m3) [Reticle LS]", K_LS_F),
    ("r_lsf_y", "Y", "Long Stroke (j=1, m3) [Reticle LS]", K_LS_F),
    ("r_ssf_x", "X", "Short Stroke (j=1, m5) [Reticle SS]", K_SS_F),
    ("r_ssf_y", "Y", "Short Stroke (j=1, m5) [Reticle SS]", K_SS_F),
    ("r_stf_x", "X", "Reticle Stage (j=1, m7) [Mask Stage]", K_ST_F),
    ("r_stf_y", "Y", "Reticle Stage (j=1, m7) [Mask Stage]", K_ST_F),
    ("mask_x",  "X", "Mask (j=1, m8) [Reticle]", K8_R),
    ("mask_y",  "Y", "Mask (j=1, m8) [Reticle]", K8_R),
]"""

# Replace the MONITOR_CONFIG block
config_pattern = re.compile(r'MONITOR_CONFIG = \[.*?\]', re.DOTALL)
content = config_pattern.sub(new_config, content)

# Replace the HDR line
content = content.replace(
    'HDR = f"{\'Stage\':<18} {\'Ax\':<2} {\'disp (mm)\':>12} {\'K (N/m)\':>12} {\'F=K*f (N)\':>14}"',
    'HDR = f"{\'Stage\':<40} {\'Ax\':<2} {\'disp (mm)\':>12} {\'K (N/m)\':>12} {\'F=K*f (N)\':>14}"'
)

# Replace the SEP line
content = content.replace('SEP = "-" * 70', 'SEP = "-" * 95')

# Replace the print line formatting
content = content.replace(
    'lines.append(f"{label:<18} {ax:<2} {disp_m*1000:>12.6f} {k_val:>12.2e} {force:>14.4f}")',
    'lines.append(f"{label:<40} {ax:<2} {disp_m*1000:>12.6f} {k_val:>12.2e} {force:>14.4f}")'
)

# Replace Lens Element check
content = content.replace(
    'label = s_name if ax == "X" or s_name == "Lens Element" else ""',
    'label = s_name if ax == "X" or "Lens element" in s_name else ""'
)

with open('litho_sim.py', 'w') as f:
    f.write(content)
