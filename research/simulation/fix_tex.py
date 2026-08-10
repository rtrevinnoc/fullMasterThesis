import re

with open('main.tex', 'r') as f:
    lines = f.readlines()

out_lines = []
for i, line in enumerate(lines):
    if i < 599:
        out_lines.append(line)
    else:
        # 1. Remove ^f from C^f and K^f
        line = line.replace('C^f', 'C')
        line = line.replace('K^f', 'K')
        
        # 2 & 3. Remove _\theta and capitalize c->C, k->K
        line = line.replace('c_{\\theta,', 'C_{')
        line = line.replace('k_{\\theta,', 'K_{')
        
        out_lines.append(line)

with open('main_fixed.tex', 'w') as f:
    f.writelines(out_lines)

print("Done. Let's see the diff:")
import os
os.system('diff -u main.tex main_fixed.tex')
