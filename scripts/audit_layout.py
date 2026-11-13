# -*- coding: utf-8 -*-
"""建筑/操场布局对账：我们的最终楼栋 vs 官方规划图逐像素差异。

A 缺失: 规划图建筑候选掩码中，未被最终楼栋覆盖的大块区域 → 漏提取的楼
B 错楼: 最终楼栋足迹下，规划图色彩以操场/球场/绿地为主的 → 误判成楼
输出 JSON 报告供修正使用。
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
OUT = os.path.join(ROOT, "data", "layout_audit.json")

WORK_W = 1800
PX_PER_M = 1.81


def to_px(x, z, h):
    return (x * PX_PER_M + WORK_W / 2, z * PX_PER_M + h / 2)


def main():
    img = cv2.imread(PLAN, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    h = int(H * WORK_W / W)
    small = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_AREA)
    b, g, r = [small[:, :, i].astype(np.int16) for i in range(3)]

    # --- 规划图侧：建筑候选掩码（与 extract_outlines 同口径）---
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from extract_outlines import building_union_mask
    union, _ = building_union_mask(small)

    # --- 我们侧：最终楼栋栅格化 ---
    d = json.load(open(OUTLINES, encoding="utf-8"))
    blds = [p for p in d["polygons"] if p["class"] == "building"]
    kept = np.zeros(small.shape[:2], np.uint8)
    for p in blds:
        pts = np.array([[int(q[0] * PX_PER_M + WORK_W / 2),
                         int(q[1] * PX_PER_M + h / 2)] for q in p["points"]], np.int32)
        cv2.fillPoly(kept, [pts], 255)

    missing = cv2.bitwise_and(union, cv2.bitwise_not(kept))
    n, labels, stats, cent = cv2.connectedComponentsWithStats(missing, 8)
    miss_list = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 900:  # <275 m² 的缺口忽略
            continue
        bx, by, bw, bh = stats[i, :4]
        if bx > 0.70 * WORK_W and by > 0.66 * h:
            continue
        if bx < 0.24 * WORK_W and by > 0.84 * h:
            continue
        px, py = cent[i]
        x, z = (px - WORK_W / 2) / PX_PER_M, (py - h / 2) / PX_PER_M
        miss_list.append({"area_px": int(a), "area_m2": round(a / PX_PER_M ** 2),
                          "world": [round(x), round(z)],
                          "bbox_px": [int(bx), int(by), int(bw), int(bh)]})
    miss_list.sort(key=lambda m: -m["area_px"])

    # --- B: 错楼判定（足迹下规划图色彩构成）---
    salmon = (np.abs(b - 128) + np.abs(g - 168) + np.abs(r - 237)) < 90
    white = ((np.minimum(np.minimum(b, g), r) > 205) &
             (np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r) < 26))
    field = ((r > g + 30) & (r > b + 30)) | \
            ((g > r + 15) & (g > b + 15) & (np.maximum(np.maximum(b, g), r) < 170))
    wrong = []
    for p in blds:
        pts = np.array([[int(q[0] * PX_PER_M + WORK_W / 2),
                         int(q[1] * PX_PER_M + h / 2)] for q in p["points"]], np.int32)
        m = np.zeros(small.shape[:2], np.uint8)
        cv2.fillPoly(m, [pts], 1)
        tot = int(m.sum())
        if tot < 200:
            continue
        s = int(salmon[m > 0].sum())
        w = int(white[m > 0].sum())
        f = int(field[m > 0].sum())
        bld_like = (s + w) / tot
        fld_like = f / tot
        cx = p["centroid"][0]
        cz = p["centroid"][1]
        if fld_like > 0.5 and bld_like < 0.25:
            wrong.append({"world": [round(cx), round(cz)],
                          "bld_like": round(bld_like, 2), "field_like": round(fld_like, 2),
                          "area_m2": round(abs(cv2.contourArea(pts)) / PX_PER_M ** 2)})
    wrong.sort(key=lambda w: -w["area_m2"])

    json.dump({"missing": miss_list, "wrong_as_building": wrong},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("缺失候选(>=275m²):", len(miss_list))
    for m in miss_list[:15]:
        print("  缺 world(%4d,%4d) %6dm² bbox=%s" % (m["world"][0], m["world"][1],
                                                     m["area_m2"], m["bbox_px"]))
    print("错楼候选(field>50%):", len(wrong))
    for w_ in wrong[:12]:
        print("  错 world(%4d,%4d) %6dm² bld%%=%.2f field%%=%.2f" %
              (w_["world"][0], w_["world"][1], w_["area_m2"], w_["bld_like"], w_["field_like"]))
    print("报告 -> data/layout_audit.json")


if __name__ == "__main__":
    main()
