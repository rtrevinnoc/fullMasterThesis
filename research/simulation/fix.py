import re

with open('litho_sim.py', 'r') as f:
    content = f.read()

# Move get_wafer_dies above MODEL_XML
func_pattern = re.compile(r'(def get_wafer_dies.*?return sorted_dies\n)', re.DOTALL)
match = func_pattern.search(content)
func_str = match.group(1)

content = content.replace(func_str, '')

# Insert it right before MODEL_XML
model_xml_pos = content.find('MODEL_XML = f"""')
insertion = func_str + """
die_grid_xml = ""
for cx, cy in get_wafer_dies(WAFER_R, DIE_L):
    die_grid_xml += f'                                    <geom type="box" size="{DIE_L/2 - 0.0005} {DIE_L/2 - 0.0005} 0.0011" pos="{cx} {cy} 0.001" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>\\n'

"""

content = content[:model_xml_pos] + insertion + content[model_xml_pos:]

# Replace wafer body content to use die_grid_xml
wafer_old = """                                <body name="wafer" pos="0 0 0.025">
                                    <joint name="wafer_x" type="slide" axis="1 0 0" stiffness="{K8_W}" damping="{C8_W}"/>
                                    <joint name="wafer_y" type="slide" axis="0 1 0" stiffness="{K8_W}" damping="{C8_W}"/>
                                    <geom type="cylinder" size="0.15 0.001" material="silicon" mass="{M_WAFR}"/>
                                    <!-- Highlight the active die being scanned -->
                                    <geom name="active_die" type="box" size="{DIE_L/2} {DIE_L/2} 0.002" pos="0 0 0.002" rgba="1 1 0 0.4" contype="0" conaffinity="0"/>
                                </body>"""
wafer_new = """                                <body name="wafer" pos="0 0 0.025">
                                    <joint name="wafer_x" type="slide" axis="1 0 0" stiffness="{K8_W}" damping="{C8_W}"/>
                                    <joint name="wafer_y" type="slide" axis="0 1 0" stiffness="{K8_W}" damping="{C8_W}"/>
                                    <geom type="cylinder" size="0.15 0.001" material="silicon" mass="{M_WAFR}"/>
{die_grid_xml}                                </body>"""
content = content.replace(wafer_old, wafer_new)

# Add indicator to lens
lens_old = """                    <!-- Lens Element (m4=0.3kg) -->
                    <body name="lens" pos="0 0 -0.4">
                        <joint name="lens_z" type="slide" axis="0 0 1" stiffness="{K4_O}" damping="{C4_O}"/>
                        <geom type="cylinder" size="0.1 0.05" material="glass" mass="{M_LENS}"/>
                    </body>"""
lens_new = """                    <!-- Lens Element (m4=0.3kg) -->
                    <body name="lens" pos="0 0 -0.4">
                        <joint name="lens_z" type="slide" axis="0 0 1" stiffness="{K4_O}" damping="{C4_O}"/>
                        <geom type="cylinder" size="0.1 0.05" material="glass" mass="{M_LENS}"/>
                        <!-- Die indicator fixed to the optics chain -->
                        <geom name="die_indicator" type="box" size="{DIE_L/2} {DIE_L/2} 0.002" pos="0 0 0.015" rgba="1 1 0 0.6" contype="0" conaffinity="0"/>
                    </body>"""
content = content.replace(lens_old, lens_new)

# Remove the line updating active_die in the loop
loop_old = """            # Update visual marker ONLY when the die index changes
            cx, cy = controller.dies[controller.die_idx]
            model.geom('active_die').pos[:] = [cx, cy, 0.002]"""
loop_new = """            # Wafer moves, indicator stays fixed to optics chain
            cx, cy = controller.dies[controller.die_idx]"""
content = content.replace(loop_old, loop_new)

# Remove the setup line
setup_old = """    # Set initial active die position on the wafer
    cx0, cy0 = dies[0]
    model.geom('active_die').pos[:] = [cx0, cy0, 0.002]"""
content = content.replace(setup_old, "")

with open('litho_sim.py', 'w') as f:
    f.write(content)

