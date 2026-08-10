import numpy as np

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

dies = get_wafer_dies(0.150, 0.01225)
print("Total dies:", len(dies))
for i in range(10):
    print(f"Die {i}: {dies[i]}")
