"""Corrected yield-aware cost function.

J_map = (1 - P_none) + P_ood
  where P_ood is the total probability mass on classes that scanner motion
  cannot plausibly cause (Center, Donut, Edge-Ring, Loc, Near-full). The
  first term prices any predicted defect; the second penalizes maps the
  classifier reads as out-of-domain morphologies, closing the blind spot
  where saturated maps concentrate mass on non-plausible classes and a
  plausible-classes-only cost would evaluate to zero.

J_total = J_map + BETA * max(0, T_wafer - T_BUDGET) / T_BUDGET
  soft throughput constraint on the wafer cycle time implied by the
  trajectory parameters.
"""
import os
import sys

import numpy as np
import torch
import cv2

CNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../nn/cnn"))
if CNN_DIR not in sys.path:
    sys.path.append(CNN_DIR)
from model import WaferCNN  # noqa: E402

LABELS = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
          'Loc', 'Near-full', 'Random', 'Scratch']
PLAUSIBLE = {'none', 'Scratch', 'Edge-Loc', 'Random'}
BETA = 4.0

_cnn = None


def cnn():
    global _cnn
    if _cnn is None:
        _cnn = WaferCNN(num_classes=9)
        _cnn.load_state_dict(torch.load(os.path.join(CNN_DIR, "wafer_cnn.pth"),
                                        map_location="cpu"))
        _cnn.eval()
    return _cnn


def classify(wafer_map):
    """Return (probs ndarray[9], top label, top prob)."""
    resized = cv2.resize(wafer_map.astype(np.float32), (64, 64),
                         interpolation=cv2.INTER_NEAREST)
    with torch.no_grad():
        logits = cnn()(torch.from_numpy(resized)[None, None])
        probs = torch.softmax(logits, dim=1)[0].numpy()
    k = int(probs.argmax())
    return probs, LABELS[k], float(probs[k])


def map_cost(wafer_map):
    """J_map = (1 - P_none) + P_out-of-domain."""
    probs, label, conf = classify(wafer_map)
    p_none = float(probs[LABELS.index('none')])
    p_ood = float(sum(probs[LABELS.index(c)] for c in LABELS
                      if c not in PLAUSIBLE))
    return (1.0 - p_none) + p_ood, dict(label=label, conf=conf,
                                        p_none=p_none, p_ood=p_ood)


def total_cost(wafer_map, t_wafer, t_budget):
    j_map, info = map_cost(wafer_map)
    penalty = BETA * max(0.0, t_wafer - t_budget) / t_budget
    info.update(j_map=j_map, t_wafer=t_wafer, penalty=penalty)
    return j_map + penalty, info


def dies_to_map(dies, die_l, wafer_r, status):
    """Arrange per-die status (1 pass / 2 fail) on the WM-811K-style grid."""
    n = int(np.ceil(2 * wafer_r / die_l)) + 2
    wm = np.zeros((2 * n, 2 * n), dtype=int)
    index = {d: k for k, d in enumerate(dies)}
    for i in range(-n, n):
        for jj in range(-n, n):
            cx, cy = i * die_l, jj * die_l
            corners = [(cx - die_l / 2, cy - die_l / 2), (cx + die_l / 2, cy - die_l / 2),
                       (cx - die_l / 2, cy + die_l / 2), (cx + die_l / 2, cy + die_l / 2)]
            if all(np.sqrt(x ** 2 + y ** 2) < wafer_r for x, y in corners):
                k = index.get((cx, cy))
                if k is not None and k < len(status):
                    wm[jj + n, i + n] = status[k]
    return wm
