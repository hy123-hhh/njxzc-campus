# -*- coding: utf-8 -*-
"""布局修正 v3（2026-11-13，官方规划图对账 + 用户红圈标注）。

规划图实际配色：新建建筑=蓝色、保留建筑=白/暖灰斜线顶、宿舍=salmon、
体育馆=浅粉、操场跑道=蓝环+浅绿球场。据此定义建筑色板：

  blue        蓝 (b>r+30 & b>g+20)                 —— 新建建筑楼体
  pale        浅 (mn>185 & sat<50)                 —— 保留建筑浅墙/浅粉体育馆
  salmon      砖红 (与 237,168,128 距<100)          —— 宿舍
  hatch_solid 暖灰斜线顶实心化 (mn130-210,sat<35 再 open15 去路面)
  darkline    深灰斜线 (mn40-115, mx<130, open3)

规则（逐楼）：
  删除  绿茵占比 pitch>0.25（操场：北运动场 10933m² 巨楼=蓝环+绿茵）
  保留  其余（bld 占比过低的仅记录待查）
补缺  bld_color 掩码 - 已保留楼 = 缺口 blob（>=900px²、非图例区、
  blob 内 pitch<0.30 防操场白线伪影）→ close15 → 轮廓筛选 → 新楼
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
OUTLINES = os.path.join(ROOT, "data", "outlines.json")

WORK_W = 1800
PX_PER_M = 1.81
FIELD_CENTERS = [(300, -450, 130), (100, 430, 130)]  # 北/南运动场中心与半径
POND_CENTERS = [(-283, 210), (-98, -106)]  # 西南池塘/中央水塘（勿建楼）


def in_ponds(x, z):
    return any((x - px) ** 2 + (z - pz) ** 2 < 55 ** 2 for px, pz in POND_CENTERS)


def poly_area(pts):
    a = 0
    for i in range(len(pts)):
        q = pts[(i + 1) % len(pts)]
        a += pts[i][0] * q[1] - q[0] * pts[i][1]
    return abs(a) / 2


def in_fields(x, z):
    return any((x - fx) ** 2 + (z - fz) ** 2 < fr ** 2 for fx, fz, fr in FIELD_CENTERS)


def main():
    img = cv2.imread(PLAN, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    h = int(H * WORK_W / W)
    small = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_AREA)
    PYC = h / 2.0
    import math
    b, g, r = [small[:, :, i].astype(np.int16) for i in range(3)]
    mn = np.minimum(np.minimum(b, g), r)
    mx = np.maximum(np.maximum(b, g), r)
    sat = mx - mn

    blue = ((b > r + 30) & (b > g + 20)).astype(np.uint8)
    pale = ((mn > 185) & (sat < 50)).astype(np.uint8)
    salmon = ((np.abs(b - 128) + np.abs(g - 168) + np.abs(r - 237)) < 100).astype(np.uint8)
    warm = ((mn > 130) & (mn < 210) & (sat < 35) & (r >= g)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    dark = ((mn > 40) & (mn < 115) & (mx < 130) & (sat < 70) &
            (~((g > r + 15) & (g > b + 15)))).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    pitch = ((g > 150) & (g > r + 25) & (g > b + 25)).astype(np.uint8)

    bld_color = np.maximum(np.maximum(np.maximum(blue, pale),
                                      np.maximum(salmon, warm)), dark)

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from extract_outlines import building_union_mask
    union, _ = building_union_mask(small)

    d = json.load(open(OUTLINES, encoding="utf-8"))
    blds = [p for p in d["polygons"] if p["class"] == "building"]
    others = [p for p in d["polygons"] if p["class"] != "building"]

    campus = np.zeros(small.shape[:2], np.uint8)
    for p in d["polygons"]:
        pts = np.array([[int(q[0] * PX_PER_M + WORK_W / 2),
                         int(q[1] * PX_PER_M + h / 2)] for q in p["points"]], np.int32)
        cv2.fillPoly(campus, [pts], 255)
    campus = cv2.dilate(campus, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 61)))

    kept = np.zeros(small.shape[:2], np.uint8)
    new_blds, removed, lowbld, added = [], [], [], []
    for p in blds:
        pts = np.array([[int(q[0] * PX_PER_M + WORK_W / 2),
                         int(q[1] * PX_PER_M + h / 2)] for q in p["points"]], np.int32)
        m = np.zeros(small.shape[:2], np.uint8)
        cv2.fillPoly(m, [pts], 1)
        tot = int(m.sum())
        cx, cz = p["centroid"][0], p["centroid"][1]
        drop = False
        if tot >= 200:
            pf = pitch[m > 0].mean()
            if pf > 0.25 and in_fields(cx, cz):
                removed.append({"world": [round(cx), round(cz)],
                                "area_m2": round(poly_area(p["points"])), "why": "操场"})
                drop = True
            else:
                bl = bld_color[m > 0].mean()
                if bl < 0.10 and poly_area(p["points"]) < 600:
                    removed.append({"world": [round(cx), round(cz)],
                                    "area_m2": round(poly_area(p["points"])), "why": "铺装碎块"})
                    drop = True
                elif bl < 0.12:
                    lowbld.append({"world": [round(cx), round(cz)],
                                   "area_m2": round(poly_area(p["points"])), "bld": round(bl, 3)})
        if not drop:
            new_blds.append(p)
            kept = cv2.bitwise_or(kept, cv2.dilate(m, np.ones((5, 5), np.uint8)))

    missing = cv2.bitwise_and(bld_color, cv2.bitwise_not(kept))
    missing = cv2.bitwise_and(missing, cv2.bitwise_not(pitch))
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    n, labels, stats, cent = cv2.connectedComponentsWithStats(missing, 8)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 900:
            continue
        bx, by, bw, bh = stats[i, :4]
        if bx > 0.70 * WORK_W and by > 0.66 * h:
            continue
        if bx < 0.24 * WORK_W and by > 0.84 * h:
            continue
        px, py = cent[i]
        x, z = (px - WORK_W / 2) / PX_PER_M, (py - h / 2) / PX_PER_M
        if in_fields(x, z) or in_ponds(x, z):
            continue
        if a > 39300 or campus[min(h - 1, int(py)), min(WORK_W - 1, int(px))] == 0:
            continue  # >12000m2 的合并巨块或校园范围外：跳过
        pad = 12
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(WORK_W, bx + bw + pad), min(h, by + bh + pad)
        loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        loc[(labels[y0:y1, x0:x1] == i)] = 255
        loc = cv2.morphologyEx(loc, cv2.MORPH_CLOSE, k15)
        loc = cv2.morphologyEx(loc, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cs, _ = cv2.findContours(loc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs_all = list(cs)
        if int(loc.sum()) > 8000 and cs_all:
            # 大 blob（多栋楼群）：腐蚀分离单栋再逐栋收
            k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            er = cv2.erode(loc, k7, iterations=2)
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(er, 8)
            cs_all = []
            for j in range(1, n2):
                if st2[j, cv2.CC_STAT_AREA] < 400:
                    continue
                comp = (lab2 == j).astype(np.uint8) * 255
                comp = cv2.dilate(comp, k7, iterations=2)
                comp = cv2.bitwise_and(comp, loc)
                c2, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cs_all.extend(c2)
        for c in cs_all:
            ar = cv2.contourArea(c)
            if ar < 700:
                continue
            bx2, by2, bw2, bh2 = cv2.boundingRect(c)
            if ar / max(bw2 * bh2, 1) < 0.40 or bw2 / max(bh2, 1) > 14 or bh2 / max(bw2, 1) > 14:
                continue
            tmp = np.zeros(loc.shape, np.uint8)
            cv2.drawContours(tmp, [c], -1, 1, -1)
            if float(tmp.sum()) / max(ar, 1) < 0.5:
                continue
            poly = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(poly) < 4:
                continue
            wpts = [[round((px2 + x0 - WORK_W / 2) / PX_PER_M, 2),
                     round((py2 + y0 - h / 2) / PX_PER_M, 2)] for px2, py2 in poly]
            cxa = sum(q[0] for q in wpts) / len(wpts)
            cza = sum(q[1] for q in wpts) / len(wpts)
            if in_fields(cxa, cza) or in_ponds(cxa, cza):
                continue
            new_blds.append({"class": "building", "points": wpts,
                             "centroid": [round(cxa, 1), round(cza, 1)],
                             "area_px": int(ar)})
            added.append({"world": [round(cxa), round(cza)],
                          "area_m2": round(ar / PX_PER_M ** 2), "npts": len(wpts)})


    # ---- 教学实验楼区规整化 v3（2026-11-21 固化：空心回字形）----
    # R1 高清实证（_b025_fix/ind_*.png、ring_overlay.png 可复核）：
    # 两簇完全相同，每簇 = 主楼(6F,回字形环带+大中庭) + 翼楼(回字形环带+长条中庭)；
    # 底部簇朝向 -9°、顶部簇 -33°（扇形布局）；用户明确要求：中间空心、四栋同形。
    # 中庭实测：主楼 29x15m（两主楼取一致，main_top 首测吸入树影弃用）、翼楼 21x6m，均居中。
    # 实现为每栋拆 4 段矩形带（上/下带全宽，左/右带补齐），中庭露空。
    WORK = 1800.0

    RING_BUILDS = [
        # (字母, cx, cy, rw, rh, ang_deg, 中庭cw, 中庭ch, 层数)  —— 工作图(1800宽)px 实测
        ("a", 960.5, 854.0, 71.5, 68.8, -32.9, 52.5, 27.2, 6),   # 顶部主楼 6F
        ("b", 907.6, 887.2, 62.5, 40.4, -33.0, 37.3, 10.6, None),  # 顶部翼楼
        ("c", 998.9, 959.6, 73.9, 69.0, -9.0, 52.5, 27.2, 6),   # 底部主楼 6F
        ("d", 939.7, 969.1, 63.4, 40.6, -8.9, 37.3, 10.6, None),   # 底部翼楼
    ]
    TEACH_ANNEX = ("building_025e", 1027.1, 889.3, 28.2, 23.2, -20.6)  # 右上配楼 2F（实心）
    TEACH_CIRCLE = ("building_025f", 994.9, 1049.2, 20.4)              # 圆形楼 2F（实心）

    teach = []
    for letter, cx, cy, rw, rh, ang, cw, ch, _fl in RING_BUILDS:
        a2, b2 = rw / 2.0, rh / 2.0
        ac, bc = cw / 2.0, ch / 2.0
        t = math.radians(ang)
        Rm = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        bands = [
            [(-a2, -b2), (a2, -b2), (a2, -bc), (-a2, -bc)],
            [(-a2, bc), (a2, bc), (a2, b2), (-a2, b2)],
            [(-a2, -bc), (-ac, -bc), (-ac, bc), (-a2, bc)],
            [(ac, -bc), (a2, -bc), (a2, bc), (ac, bc)],
        ]
        for k, bd in enumerate(bands, 1):
            pts = (np.array(bd) @ Rm.T) + [cx, cy]
            wpts = [[round(float((px - WORK_W / 2) / PX_PER_M), 2),
                     round(float((py - PYC) / PX_PER_M), 2)] for px, py in pts]
            teach.append({"id": f"building_025{letter}{k}", "class": "building",
                          "points": wpts,
                          "centroid": [round(float(np.mean([q[0] for q in wpts])), 1),
                                       round(float(np.mean([q[1] for q in wpts])), 1)],
                          "area_px": round(float(abs(cv2.contourArea(
                              np.array(wpts, np.float32).reshape(-1, 1, 2)))), 1)})
    bid_e, ecx, ecy, erw, erh, eang = TEACH_ANNEX
    eb = cv2.boxPoints(((ecx, ecy), (erw, erh), eang))
    wpts = [[round(float((px - WORK_W / 2) / PX_PER_M), 2),
             round(float((py - PYC) / PX_PER_M), 2)] for px, py in eb]
    teach.append({"id": bid_e, "class": "building", "points": wpts,
                  "centroid": [round(float(np.mean([q[0] for q in wpts])), 1),
                               round(float(np.mean([q[1] for q in wpts])), 1)],
                  "area_px": round(float(abs(cv2.contourArea(eb.reshape(-1, 1, 2)))), 1)})
    bid_c2, ccx, ccy, cr = TEACH_CIRCLE
    angs = np.linspace(0, 2 * math.pi, 21)[:-1]
    wpts = [[round(float((ccx + cr * math.cos(a2v) - WORK_W / 2) / PX_PER_M), 2),
             round(float((ccy + cr * math.sin(a2v) - PYC) / PX_PER_M), 2)] for a2v in angs]
    teach.append({"id": bid_c2, "class": "building", "points": wpts,
                  "centroid": [round(float((ccx - WORK_W / 2) / PX_PER_M), 1),
                               round(float((ccy - PYC) / PX_PER_M), 1)],
                  "area_px": round(float(3.14159 * cr * cr), 1)})
    n025_before = len([p for p in new_blds if str(p.get("id", "")).startswith("building_025")])
    new_blds = [p for p in new_blds if not str(p.get("id", "")).startswith("building_025")]
    new_blds.extend(teach)
    print("教学区规整化v3: 删旧025系", n025_before, "栋, 写入空心回字形",
          len(RING_BUILDS), "栋x4段 + 配楼 + 圆楼 =", len(teach), "个多边形")

    # ---- 南群规整化（2026-11-21 固化：陶行知像东北两对主楼+翼楼，同空心回字形）----
    # R1 高清实证（_b025_fix/south_ring_overlay2.png、grid_annex2.png 可复核）：
    # 两对组合：翼楼n+主楼n 同朝向 29.1°，翼楼w+主楼s 同朝向 53.1°（扇形布局继续旋转）；
    # 尺寸与教学区同款（翼楼35x21、主楼41x37、中庭 20x5 / 29x15）；2F配楼 18x13m。
    # 替换该区域 5 个无 id 毛边多边形 + building_042（配楼区重复多边形）。
    RING_BUILDS_S = [
        # (字母, cx, cy, rw, rh, ang_deg, 中庭cw, 中庭ch, 中庭偏移dx, 中庭偏移dy, 层数)
        ("g", 912.4, 1101.0, 63.5, 38.3, 29.1, 35.0, 9.5, 10.0, 0.4, None),   # 北翼楼
        ("h", 965.7, 1134.1, 70.6, 66.6, 29.1, 50.7, 30.0, 6.4, -1.0, 6),     # 北主楼 6F
        ("i", 855.3, 1166.3, 63.6, 36.4, 53.1, 35.0, 9.5, 10.0, 0.4, None),   # 西翼楼
        ("j", 893.3, 1216.6, 74.0, 64.8, 53.1, 50.7, 30.0, 6.4, -1.0, 6),     # 南主楼 6F
    ]
    SOUTH_ANNEX = ("building_025k", 970.3, 1211.0, 32.0, 24.0, 41.9)  # 2F配楼（实心）
    SOUTH_ZONE = (-28, 62, -50, 25)   # x0,x1,y0,y1：无id毛边多边形清除区

    xz0, xz1, yz0, yz1 = SOUTH_ZONE

    def _in_zone(p):
        if p.get("id"):
            return False
        cxx, czz = p["centroid"][0], p["centroid"][1]
        return xz0 <= cxx <= xz1 and yz0 <= czz <= yz1

    n_south_before = len([p for p in new_blds if _in_zone(p)])
    new_blds = [p for p in new_blds if not _in_zone(p)]
    new_blds = [p for p in new_blds if p.get("id") != "building_042"]
    for letter, cx, cy, rw, rh, ang, cw, ch, odx, ody, _fl in RING_BUILDS_S:
        a2, b2 = rw / 2.0, rh / 2.0
        ac, bc = cw / 2.0, ch / 2.0
        t = math.radians(ang)
        Rm = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        bands = [
            [(-a2, -b2), (a2, -b2), (a2, -bc), (-a2, -bc)],
            [(-a2, bc), (a2, bc), (a2, b2), (-a2, b2)],
            [(-a2, -bc), (-ac, -bc), (-ac, bc), (-a2, bc)],
            [(ac, -bc), (a2, -bc), (a2, bc), (ac, bc)],
        ]
        for k, bd in enumerate(bands, 1):
            pts = (np.array(bd) @ Rm.T) + [cx + odx, cy + ody]
            wpts = [[round(float((px - WORK_W / 2) / PX_PER_M), 2),
                     round(float((py - PYC) / PX_PER_M), 2)] for px, py in pts]
            new_blds.append({"id": f"building_025{letter}{k}", "class": "building",
                             "points": wpts,
                             "centroid": [round(float(np.mean([q[0] for q in wpts])), 1),
                                          round(float(np.mean([q[1] for q in wpts])), 1)],
                             "area_px": round(float(abs(cv2.contourArea(
                                 np.array(wpts, np.float32).reshape(-1, 1, 2)))), 1)})
    bid_k, kcx, kcy, kw, kh, kang = SOUTH_ANNEX
    kb = cv2.boxPoints(((kcx, kcy), (kw, kh), kang))
    wpts = [[round(float((px - WORK_W / 2) / PX_PER_M), 2),
             round(float((py - PYC) / PX_PER_M), 2)] for px, py in kb]
    new_blds.append({"id": bid_k, "class": "building", "points": wpts,
                     "centroid": [round(float(np.mean([q[0] for q in wpts])), 1),
                                  round(float(np.mean([q[1] for q in wpts])), 1)],
                     "area_px": round(float(abs(cv2.contourArea(kb.reshape(-1, 1, 2)))), 1)})
    print("南群规整化: 清除无id毛边多边形", n_south_before, "个 + building_042, 写入环带",
          len(RING_BUILDS_S), "栋x4段 + 配楼 =", len(RING_BUILDS_S) * 4 + 1, "个多边形")

    # ---- 全校误楼清除 + 西侧板楼规整化（2026-11-21 固化，audit_sheet1/2 对照核验）----
    # 补缺步骤曾把树影花圃/农田/停车场/马路误识别为楼，逐格对照规划图确认后清除：
    FALSE_BLD = [  # (世界cx, cz) 误楼质心，就近<12m 删除
        (-143, -211),  # 花圃梯田
        (-31, -256),   # 树木/花园
        (232, -168),   # 铺装/花园
        (394, -647),   # 马路
        (-162, -166),  # 农田花圃
        (283, 72),     # 树木/停车场
        (-22, 52),     # 农田花圃
        (-190, 283),   # 停车场/田径场边
    ]
    n_false = len([p for p in new_blds
                   if any(abs(p["centroid"][0] - fx) < 12 and abs(p["centroid"][1] - fz) < 12
                          for fx, fz in FALSE_BLD)])
    new_blds = [p for p in new_blds
                if not any(abs(p["centroid"][0] - fx) < 12 and abs(p["centroid"][1] - fz) < 12
                           for fx, fz in FALSE_BLD)]
    print("误楼清除:", n_false, "栋(花圃/树木/马路/停车场误识别)")

    # 西侧两栋5F板楼（原无id毛边连体多边形，图上标5F）拆为两栋规整矩形
    WEST_SLABS = [
        ("building_025l1", 583.4, 1149.3, 57.6, 18.3, 21.3),   # 北板楼 5F
        ("building_025l2", 568.0, 1184.2, 47.7, 19.8, 22.8),   # 南板楼 5F
    ]
    n_west_before = len([p for p in new_blds
                         if abs(p["centroid"][0] - (-179)) < 15 and abs(p["centroid"][1] - (-16)) < 20])
    new_blds = [p for p in new_blds
                if not (abs(p["centroid"][0] - (-179)) < 15 and abs(p["centroid"][1] - (-16)) < 20
                        and not p.get("id"))]
    for bid, cx, cy, rw, rh, ang in WEST_SLABS:
        b = cv2.boxPoints(((cx, cy), (rw, rh), ang))
        wpts = [[round(float((px - WORK_W / 2) / PX_PER_M), 2),
                 round(float((py - PYC) / PX_PER_M), 2)] for px, py in b]
        new_blds.append({"id": bid, "class": "building", "points": wpts,
                         "centroid": [round(float(np.mean([q[0] for q in wpts])), 1),
                                      round(float(np.mean([q[1] for q in wpts])), 1)],
                         "area_px": round(float(abs(cv2.contourArea(b.reshape(-1, 1, 2)))), 1)})
    print("西侧板楼规整化: 清除毛边连体", n_west_before, "个, 写入2栋5F矩形(端翼并入主体近似)")



    json.dump({"class": "layout", "polygons": others + new_blds},
              open(OUTLINES, "w", encoding="utf-8"), ensure_ascii=False)
    print("删除操场误楼:", len(removed))
    for m in removed:
        print("  - world(%d,%d) %dm2 (%s)" % (m["world"][0], m["world"][1], m["area_m2"], m["why"]))
    print("bld 占比待查(<0.12，已保留):", len(lowbld))
    for m in lowbld[:8]:
        print("  ? world(%d,%d) %dm2 bld=%.3f" % (m["world"][0], m["world"][1], m["area_m2"], m["bld"]))
    print("新增楼栋:", len(added))
    for m in added:
        print("  + world(%4d,%4d) %6dm2 npts=%d" % (m["world"][0], m["world"][1], m["area_m2"], m["npts"]))
    print("总楼数:", len(new_blds))


if __name__ == "__main__":
    main()
