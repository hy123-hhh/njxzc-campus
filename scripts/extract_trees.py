# -*- coding: utf-8 -*-
"""从官方规划图提取树符号位置（仅本地运行）。

树符号 = 深绿圆点（g 主导、亮度低）；草坪/浅绿不命中。
输出 data/trees.json: [{"x","z","r","h"}...]，米制世界坐标。
排除：建筑轮廓内、水面、图例/指北针区、校园边界外；按 2.8m 间距去重。
"""
import json
import os

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
OUT = os.path.join(ROOT, "data", "trees.json")

WORK_W = 1800
PX_PER_M = 1.81


def main():
    img = cv2.imread(PLAN, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("读不到规划图")
    H, W = img.shape[:2]
    h = int(H * WORK_W / W)
    small = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_AREA)
    b, g, r = [small[:, :, i].astype(np.int16) for i in range(3)]
    mx = np.maximum(np.maximum(r, g), b)

    tree = ((g > r + 15) & (g > b + 15) & (mx < 165)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tree = cv2.morphologyEx(tree, cv2.MORPH_OPEN, k)
    water = ((b > r + 40) & (b > g + 25) & (b > 185)).astype(np.uint8)

    n, labels, stats, cent = cv2.connectedComponentsWithStats(tree, 8)

    d = json.load(open(OUTLINES, encoding="utf-8"))
    polys = d["polygons"]
    bld_polys = [Polygon(p["points"]).buffer(0.6) for p in polys
                 if p["class"] == "building" and len(p["points"]) >= 3]

    def to_world(px, py):
        return ((px - WORK_W / 2) / PX_PER_M, (py - h / 2) / PX_PER_M)

    cands = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (12 <= a <= 900):
            continue
        px, py = cent[i]
        u, v = int(px), int(py)
        if v > 1880 and u > 1155:      # 编号图例
            continue
        if v > 1630 and u > 1540:      # 色卡图例
            continue
        if v > 2040 and u < 320:       # 指北针/比例尺
            continue
        if water[v, u]:
            continue
        x, z = to_world(px, py)
        if abs(x) > 470 or abs(z) > 630:
            continue
        pt = Point(x, z)
        if any(p.contains(pt) for p in bld_polys):
            continue
        cands.append((a, x, z))

    # 大符号优先，2.8m 间距去重
    cands.sort(key=lambda c: -c[0])
    kept = []
    for a, x, z in cands:
        ok = True
        for _, kx, kz in kept[-60:] if len(kept) > 400 else kept:
            if (x - kx) ** 2 + (z - kz) ** 2 < 2.8 ** 2:
                ok = False
                break
        if ok:
            kept.append((a, x, z))

    rng = np.random.default_rng(20261113)
    trees = []
    for i, (a, x, z) in enumerate(kept):
        f = (i * 2654435761 % 1000) / 1000.0
        trees.append({"x": round(x, 2), "z": round(z, 2),
                      "r": round(1.5 + 1.1 * f, 2),
                      "h": round(3.2 + 2.0 * ((i * 40503 % 999) / 999.0), 2)})
    json.dump({"note": "树位置逐像素取自官方总平面图深绿树符号，含尺寸随机变化",
               "count": len(trees), "trees": trees},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("候选", len(cands), "去重后", len(trees), "棵 -> data/trees.json")


if __name__ == "__main__":
    main()
