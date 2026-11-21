# -*- coding: utf-8 -*-
"""第二阶段：为 1:1 轮廓模型生成程序化真实材质、UV、光影与离线预览。

仅云端 GitHub Actions 运行。输入 data/outlines.json，输出：
  models/campus_textured.glb   带 UV 与内嵌纹理的 glTF 2.0
  models/materials_report.json
  docs/离线预览.html            自包含材质光影预览（内嵌 6 张纹理）
  docs/materials/*.png          纹理源文件
"""
import base64
import json
import math
import os
import struct

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
MATDIR = os.path.join(ROOT, "docs", "materials")
GLB = os.path.join(ROOT, "models", "campus_textured.glb")
REPORT = os.path.join(ROOT, "models", "materials_report.json")
PREVIEW = os.path.join(ROOT, "docs", "离线预览.html")

# 徽派配色（2026-11-12 按校史馆沙盘实拍校准；v3 含渲染修复+池塘还原 2026-11-13）：粉墙近白 / 小瓦中灰 / 黛瓦压顶山墙保持深色对比
FLAT = {
    "building": (224, 226, 225), "roof": (108, 114, 120),
    "water": (111, 168, 207), "green": (127, 168, 90),
    "road": (192, 196, 188), "maotou": (34, 39, 44), "earth": (150, 148, 138),
}
# 规划图在 small(1800x2395) 像素空间下的尺寸与比例尺（与 refine_geometry 一致）
PLAN_W, PLAN_H, PX_PER_M = 1800, 2395, 1.81


def make_textures():
    os.makedirs(MATDIR, exist_ok=True)
    rng = np.random.default_rng(20261111)
    out = {}
    S = 256

    # 小瓦：中灰瓦面 + 竖向瓦垄（弧形瓦脊）——色深按沙盘实拍校准
    roof = np.full((S, S, 3), (112, 118, 124), np.uint8)
    roof = (roof.astype(np.int16) + rng.normal(0, 4, (S, S, 1))).clip(0, 255).astype(np.uint8)
    for x in range(6, S, 14):
        cv2.line(roof, (x, 0), (x, S), (86, 92, 98), 2)
        cv2.line(roof, (x + 7, 0), (x + 7, S), (140, 146, 152), 1)
    out["roof"] = roof

    # 蓝灰瓦：东侧宿舍板楼（沙盘实拍 2026-11-14）
    roof_b = np.full((S, S, 3), (140, 124, 110), np.uint8)
    roof_b = (roof_b.astype(np.int16) + rng.normal(0, 4, (S, S, 1))).clip(0, 255).astype(np.uint8)
    for x in range(6, S, 14):
        cv2.line(roof_b, (x, 0), (x, S), (118, 102, 90), 2)
        cv2.line(roof_b, (x + 7, 0), (x + 7, S), (162, 148, 134), 1)
    out["roof_blue"] = roof_b

    # 橙黄瓦：小体量建筑点缀（沙盘实拍）
    roof_w = np.full((S, S, 3), (88, 142, 196), np.uint8)
    roof_w = (roof_w.astype(np.int16) + rng.normal(0, 4, (S, S, 1))).clip(0, 255).astype(np.uint8)
    for x in range(6, S, 14):
        cv2.line(roof_w, (x, 0), (x, S), (70, 118, 168), 2)
        cv2.line(roof_w, (x + 7, 0), (x + 7, S), (108, 166, 222), 1)
    out["roof_warm"] = roof_w

    # 高层塔楼墙：浅灰底 + 横向楼板暗带（每 3.2m 一层）+ 竖向窗框
    tw = np.full((S, S, 3), (200, 204, 210), np.uint8)
    tw = (tw.astype(np.int16) + rng.normal(0, 3, (S, S, 1))).clip(0, 255).astype(np.uint8)
    tw[70:186, :] = (108, 118, 132)
    tw[70:186, :] = (tw[70:186, :].astype(np.int16) + rng.normal(0, 6, (186 - 70, S, 1))).clip(0, 255).astype(np.uint8)
    for x in range(0, S, 52):
        cv2.line(tw, (x, 70), (x, 186), (84, 94, 108), 3)
    for y in (68, 186):
        cv2.line(tw, (0, y), (S, y), (150, 156, 164), 2)
    out["tower_wall"] = tw

    # 青砖粉墙：浅底 + 顶部楼板暗带（UV 每竖向重复=一层 3.2m）+ 砖缝
    wall = np.full((S, S, 3), (208, 210, 208), np.uint8)
    wall = (wall.astype(np.int16) + rng.normal(0, 3, (S, S, 1))).clip(0, 255).astype(np.uint8)
    cv2.line(wall, (0, 6), (S, 6), (150, 155, 162), 10)
    cv2.line(wall, (0, 18), (S, 18), (190, 193, 196), 3)
    bh = 18
    for y in range(26, S, bh):
        cv2.line(wall, (0, y), (S, y), (172, 176, 174), 2)
        off = 0 if (y // bh) % 2 == 0 else 14
        for x in range(off, S, 28):
            cv2.line(wall, (x, y), (x, y + bh), (172, 176, 174), 1)
    out["wall"] = wall

    glass = np.full((S, S, 3), (206, 179, 143), np.uint8)
    for x in range(0, S, 20):
        cv2.line(glass, (x, 0), (x, S), (170, 150, 125), 1)
    for y in range(0, S, 20):
        cv2.line(glass, (0, y), (S, y), (170, 150, 125), 1)
    cv2.line(glass, (0, S), (S, 0), (235, 220, 200), 12)
    out["glass"] = glass

    water = np.full((S, S, 3), (207, 168, 111), np.uint8)
    for y in range(0, S, 9):
        pts = [(x, int(y + 3 * math.sin(x / 22 + y / 7))) for x in range(0, S, 4)]
        cv2.polylines(water, [np.array(pts, np.int32)], False, (225, 196, 150), 1)
    out["water"] = water

    green = np.full((S, S, 3), (90, 168, 127), np.uint8)
    for _ in range(240):
        x, y = int(rng.integers(0, S)), int(rng.integers(0, S))
        r = int(rng.integers(2, 7))
        cv2.circle(green, (x, y), r, (70, 130, 95) if r > 4 else (130, 190, 110), -1)
    out["green"] = green

    road = np.full((S, S, 3), (192, 200, 201), np.uint8)
    for x in range(0, S, 32):
        cv2.line(road, (x, 0), (x, S), (170, 174, 172), 1)
    for y in range(0, S, 32):
        cv2.line(road, (0, y), (S, y), (170, 174, 172), 1)
    out["road"] = road

    for name, arr in out.items():
        cv2.imwrite(os.path.join(MATDIR, name + ".png"), arr)
    return {k: cv2.imencode(".png", v)[1].tobytes() for k, v in out.items()}


def load_ground():
    """读取官方规划图并做成 3D 地面贴图（2026-11-12 按校史馆沙盘实拍校准）：
    ① 擦除图例/指北针/比例尺区（沙盘上没有这些）；
    ② 细红线条（用地红线、编号圆点）去除——形态学保大去小，操场跑道大色块不受影响；
    ③ 纸面白 → 草坪绿（亮度缩放），模拟沙盘草坪底色；灰路/黄广场/蓝湖/绿带原样保留。"""
    p = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return b""
    small = cv2.resize(img, (PLAN_W, PLAN_H), interpolation=cv2.INTER_AREA)
    LAWN = (142, 196, 171)  # BGR
    # ① 图例/指北针擦除（坐标为 1800x2395 工作像素，均在校园轮廓外）
    small[1880:PLAN_H, 1155:1800] = LAWN   # 编号图例两列
    small[1630:1880, 1540:1800] = LAWN     # 色卡图例
    small[2040:PLAN_H, 0:320] = LAWN       # 指北针+比例尺
    # ② 细红线条去除（保大去小：跑道大色块保留）
    b, g, r = small[:, :, 0].astype(np.int16), small[:, :, 1].astype(np.int16), small[:, :, 2].astype(np.int16)
    red = ((r > 140) & (r > g + 50) & (r > b + 50)).astype(np.uint8) * 255
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    red_big = cv2.morphologyEx(red, cv2.MORPH_OPEN, k9)
    red_thin = cv2.subtract(red, red_big)
    small[red_thin > 0] = LAWN
    # ③ 纸面白 → 草坪绿
    mn = small.min(axis=2)
    paper = mn > 210
    lum = small.mean(axis=2)
    tgt = np.clip((lum / 238.0)[..., None] * np.array([171, 196, 142], float)[None, None, :],
                  0, 255).astype(np.uint8)
    small[paper] = tgt[paper]
    # 编号红圈定点擦除（2026-11-14：23个红圈在保大去小后残留，用环带中值色填充）
    CIRCLE_ERASE = ((931, 259), (1435, 316), (645, 466), (866, 602), (1197, 606),
                    (609, 706), (1367, 751), (623, 877), (981, 912), (1352, 928),
                    (444, 936), (1591, 957), (532, 1100), (967, 1130), (1345, 1203),
                    (1318, 1385), (824, 1427), (1137, 1428), (519, 1479), (831, 1586),
                    (748, 1765), (585, 1796), (1038, 1958))
    for cx, cy in CIRCLE_ERASE:
        x0, y0 = max(0, cx - 16), max(0, cy - 16)
        x1, y1 = min(PLAN_W, cx + 16), min(PLAN_H, cy + 16)
        ring = []
        for dx in range(-30, 31, 5):
            for dy in range(-30, 31, 5):
                if abs(dx) <= 18 and abs(dy) <= 18:
                    continue
                xx, yy = cx + dx, cy + dy
                if 0 <= xx < PLAN_W and 0 <= yy < PLAN_H:
                    ring.append(small[yy, xx])
        if ring:
            small[y0:y1, x0:x1] = np.median(np.array(ring, np.uint8), axis=0).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes() if ok else b""


NAMED_BUILDINGS = [
    # 依据规划图编号图例（1-24）与红色编号圈实测定位（2026-11-14）
    ("图书馆", (-30, -339)),
    ("工科楼", (-153, -406)),
    ("文科楼", (-170, -270)),
    ("教学实验楼", (34, -143)),
    ("理科组团", (-220, 261)),
    ("行政办公楼", (-203, -33)),
    ("食堂", (259, -249)),
    ("音乐楼", (-38, 212)),
    ("美术楼", (-22, 132)),
    ("行知组团", (167, 278)),
    ("风雨操场", (-93, 317)),
    ("综合科技楼", (382, 269)),
    # 宿舍区：规划图"14号=学生宿舍"红圈×5片。只标图上实证的位置/栋数，
    # 官方A-G区名无法与图上位置一一对应（网上位置描述互相矛盾），故不标区名。
    ("宿舍① 7栋", (164, -327)),
    ("宿舍② 6栋", (250, -149)),
    ("宿舍③ 3栋", (382, -133)),
    ("宿舍④ 6栋", (246, 3)),
    ("宿舍⑤ 10栋", (131, 127)),
]


def nearest_building(polys, x, z):
    bs = [p for p in polys if p["class"] == "building"]
    return min(bs, key=lambda p: math.dist(p["centroid"], (x, z))) if bs else None


def compute_labels(polys):
    """主要功能区中文标注（按规划图编号圈实测锚定，2026-11-14）。"""
    labels = []
    for name, (x, z) in NAMED_BUILDINGS:
        b = nearest_building(polys, x, z)
        if b is not None and math.dist(b["centroid"], (x, z)) < 60:
            c = b["centroid"]
            labels.append({"n": name, "x": c[0], "z": c[1]})
    # 楼栋编号：学校官方幢号未完整查到（仅知B区14幢、E区31幢），不逐栋自编。
    waters = [p for p in polys if p["class"] == "water"]
    if waters:
        w = max(waters, key=lambda p: p.get("area_px", 0))
        labels.append({"n": "人工湖", "x": w["centroid"][0], "z": w["centroid"][1]})
    labels += [
        {"n": "陶行知像", "x": -60, "z": -30},
        {"n": "北运动场", "x": 274, "z": -478},
        {"n": "南运动场", "x": 69, "z": 427},
        {"n": "西门", "x": -400, "z": 0},
        {"n": "校名石", "x": -430, "z": 180},
    ]
    return labels


def compute_details(polys):
    """按官方总平面图的真实位置回填场景细节（新米制坐标）。"""
    d = []

    def add(t, x, z, w, dd, h, c, n):
        d.append({"t": t, "x": x, "z": z, "w": w, "d": dd, "h": h, "c": c, "n": n})

    add("plinth", -60, -30, 16, 12, 3, "#9c4a3c", "陶像红基座")
    add("statue", -60, -30, 5, 5, 13, "#a9684f", "陶行知立像")
    add("steps", -60, -14, 40, 14, 2, "#d9d8d2", "合一广场大台阶")
    add("roundbed", -60, -170, 60, 60, 1.5, "#8fae6b", "中心圆形花坛")
    waters = sorted([p for p in polys if p["class"] == "water"],
                    key=lambda p: -p.get("area_px", 0))
    if waters:
        c = waters[0]["centroid"]
        add("pavilion", c[0], c[1], 18, 18, 10, "#8c6d55", "湖心亭")
    # 图书馆双塔已移除（2026-11-15：规划图无楼层标注、沙盘实拍无并排双塔，臆造项）
    # 树列由 data/trees.json 真实树位替代（2026-11-13）
    # 花带/跑道环已由地面贴图承载（2026-11-12），3D 实心盒视觉过大，移除
    add("rockery", -400, -200, 50, 30, 5, "#b9b5a9", "西门假山")
    add("monument", -430, 180, 24, 9, 6, "#a9a49a", "校名石")
    add("bridge", 100, -20, 70, 9, 2, "#cfc9bb", "湖上石桥")
    return d


TREE_GREENS = ["#5d8a4a", "#6a975a", "#4f7d42", "#77a163"]
TREES_PATH = os.path.join(ROOT, "data", "trees.json")


def tree_geometry(t):
    """单棵树：3 面树干 + 8 棱台树冠（24 三角面/棵），四种绿色哈希分桶。"""
    x, z, r, h = t["x"], t["z"], t["r"], t["h"]
    hsh = int(abs(x) * 7 + abs(z) * 3)
    col = "#d9a441" if hsh % 8 == 7 else TREE_GREENS[hsh % 4]  # 沙盘双色树：约1/8金黄
    out = []
    m = TMesh()
    for i in range(3):
        a, b = 2 * math.pi * i / 3, 2 * math.pi * (i + 1) / 3
        q0 = (x + 0.16 * math.cos(a), z + 0.16 * math.sin(a))
        q1 = (x + 0.16 * math.cos(b), z + 0.16 * math.sin(b))
        m.quad([(q0[0], 0, q0[1]), (q1[0], 0, q1[1]),
                (q1[0], 1.1, q1[1]), (q0[0], 1.1, q0[1])],
               (0, 0, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])
    out.append((m, "solid_6b4f3a"))
    c = TMesh()
    seg = 8
    y0, y1 = 0.9, h
    r0, r1 = r, r * 0.72
    for i in range(seg):
        a, b = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
        q0 = (x + r0 * math.cos(a), z + r0 * math.sin(a))
        q1 = (x + r0 * math.cos(b), z + r0 * math.sin(b))
        p0 = (x + r1 * math.cos(a), z + r1 * math.sin(a))
        p1 = (x + r1 * math.cos(b), z + r1 * math.sin(b))
        c.quad([(q0[0], y0, q0[1]), (q1[0], y0, q1[1]),
                (p1[0], y1, p1[1]), (p0[0], y1, p0[1])],
               (0, 0, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])
        c.tri([(x, y1, z), (p0[0], y1, p0[1]), (p1[0], y1, p1[1])],
              (0, 1, 0), [(.5, .5), (0, 0), (1, 0)])
    out.append((c, "solid_" + col.lstrip("#")))
    return out


def detail_geometry(item):
    """把场景细节转成带 UV 的网格片段，返回 [(TMesh, material_key)]。"""
    t, x, z = item["t"], item["x"], item["z"]
    w, dd, h, col = item["w"], item["d"], item["h"], item["c"]
    key = "solid_" + col.lstrip("#")
    out = []

    def box_mesh(cx, cz, ww, dz, hh, base=0.0):
        m = TMesh()
        x0, x1, z0, z1 = cx - ww / 2, cx + ww / 2, cz - dz / 2, cz + dz / 2
        y0, y1 = base, base + hh
        m.quad([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
               (0, 1, 0), [(0, 0), (ww / 14, 0), (ww / 14, dz / 14), (0, dz / 14)])
        m.quad([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
               (0, 0, 1), [(0, 0), (ww / 14, 0), (ww / 14, hh / 14), (0, hh / 14)])
        m.quad([(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
               (1, 0, 0), [(0, 0), (dz / 14, 0), (dz / 14, hh / 14), (0, hh / 14)])
        m.quad([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
               (-1, 0, 0), [(0, 0), (dz / 14, 0), (dz / 14, hh / 14), (0, hh / 14)])
        return m

    def cyl_mesh(cx, cz, r, hh, base=0.0, seg=12):
        m = TMesh()
        for i in range(seg):
            a, b = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
            p0, p1 = (cx + r * math.cos(a), cz + r * math.sin(a)), (cx + r * math.cos(b), cz + r * math.sin(b))
            m.quad([(p0[0], base, p0[1]), (p1[0], base, p1[1]),
                    (p1[0], base + hh, p1[1]), (p0[0], base + hh, p0[1])],
                   (0, 0, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])
            m.tri([(cx, base + hh, cz), (p0[0], base + hh, p0[1]), (p1[0], base + hh, p1[1])],
                  (0, 1, 0), [(.5, .5), (0, 0), (1, 0)])
        return m

    def cone_mesh(cx, cz, r, hh, base, seg=10):
        m = TMesh()
        for i in range(seg):
            a, b = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
            m.tri([(cx, base + hh, cz),
                   (cx + r * math.cos(a), base, cz + r * math.sin(a)),
                   (cx + r * math.cos(b), base, cz + r * math.sin(b))],
                  (0, 1, 0), [(.5, 1), (0, 0), (1, 0)])
        return m

    if t == "statue":
        out.append((cyl_mesh(x, z, w * .28, h * .72), key))
        out.append((cyl_mesh(x, z, w * .16, h * .18, h * .72), key))
    elif t == "roundbed":
        out.append((cyl_mesh(x, z, min(w, dd) / 2, h, 0, 20), key))
    elif t == "pavilion":
        out.append((cyl_mesh(x, z, min(w, dd) / 2, h * .62, 0, 12), "solid_d8d3c6"))
        out.append((cone_mesh(x, z, min(w, dd) * .72, h * .48, h * .62), "solid_3c4a63"))
    elif t == "tower":
        out.append((box_mesh(x, z, w, dd, h), key))
        out.append((cone_mesh(x, z, min(w, dd) * .75, h * .28, h), "solid_3c4a63"))
    elif t == "treeline":
        n = max(6, int(dd / 45))
        trunk = TMesh()
        crown = TMesh()
        for i in range(n):
            zz = z - dd / 2 + dd * (i + .5) / n
            trunk.quad([(x - 1, 0, zz), (x + 1, 0, zz), (x + 1, h * .5, zz), (x - 1, h * .5, zz)],
                       (0, 0, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])
            for j in range(8):
                a, b = 2 * math.pi * j / 8, 2 * math.pi * (j + 1) / 8
                crown.tri([(x, h, zz),
                           (x + 5 * math.cos(a), h * .45, zz + 5 * math.sin(a)),
                           (x + 5 * math.cos(b), h * .45, zz + 5 * math.sin(b))],
                          (0, 1, 0), [(.5, 1), (0, 0), (1, 0)])
        out.append((trunk, "solid_8a7256"))
        out.append((crown, key))
    elif t == "track":
        for cx, cz, ww, dz in ((x, z - dd / 2, w, 5), (x, z + dd / 2, w, 5),
                               (x - w / 2, z, 5, dd), (x + w / 2, z, 5, dd)):
            out.append((box_mesh(cx, cz, ww, dz, h), key))
    elif t == "steps":
        for i in range(3):
            out.append((box_mesh(x, z + i * dd * .08, w * (1 - i * .12), dd * (1 - i * .12), h * (i + 1) / 3), key))
    else:
        out.append((box_mesh(x, z, w, dd, h), key))
    return out


class TMesh:
    def __init__(self):
        self.pos, self.nrm, self.uv, self.idx = [], [], [], []

    def quad(self, pts, normal, uvs):
        b = len(self.pos)
        for p, u in zip(pts, uvs):
            self.pos.append(p); self.nrm.append(normal); self.uv.append(u)
        self.idx.extend([b, b + 1, b + 2, b, b + 2, b + 3])

    def tri(self, pts, normal, uvs):
        b = len(self.pos)
        for p, u in zip(pts, uvs):
            self.pos.append(p); self.nrm.append(normal); self.uv.append(u)
        self.idx.extend([b, b + 1, b + 2])


def uv_top(a, b, scale=14.0):
    return (a / scale, b / scale)


def tris_of(points):
    try:
        p = Polygon(points)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.geom_type != "Polygon":
            return []
        out = []
        for t in triangulate(p):
            if p.contains(t.representative_point()):
                out.append([(float(a), float(b)) for a, b in list(t.exterior.coords)[:3]])
        return out
    except Exception:
        return []


def building_height(area):
    """按公开资料中的常见层数分档（层高约 3.3m），替代纯面积公式。"""
    if area > 9000:
        return 21.0
    if area > 3500:
        return 18.0
    if area > 1200:
        return 15.0
    return 11.0


def build_parts(polys, details=None):
    parts = []
    # 地基底板：把图纸地面“托”在一块实土色薄板上，避免建筑像悬空贴图
    slab = TMesh()
    X0, Z0, X1, Z1 = -505, -668, 505, 668
    y0, y1 = -0.6, 0.0
    slab.quad([(X0, y1, Z0), (X1, y1, Z0), (X1, y1, Z1), (X0, y1, Z1)], (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    slab.quad([(X0, y0, Z1), (X1, y0, Z1), (X1, y0, Z0), (X0, y0, Z0)], (0, -1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    slab.quad([(X0, y0, Z0), (X1, y0, Z0), (X1, y1, Z0), (X0, y1, Z0)], (0, 0, -1), [(0, 0), (1, 0), (1, 1), (0, 1)])
    slab.quad([(X1, y0, Z1), (X0, y0, Z1), (X0, y1, Z1), (X1, y1, Z1)], (0, 0, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])
    slab.quad([(X0, y0, Z0), (X0, y0, Z1), (X0, y1, Z1), (X0, y1, Z0)], (-1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    slab.quad([(X1, y0, Z1), (X1, y0, Z0), (X1, y1, Z0), (X1, y1, Z1)], (1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    parts.append((slab, "earth"))
    for p in polys:
        cls, pts = p["class"], p["points"]
        if len(pts) < 3:
            continue
        if cls != "building":
            continue  # 水/绿/路/操场由规划图贴图地面承载，不再铺平色覆盖层
        tris = tris_of(pts)
        if not tris:
            continue
        area = abs(Polygon(pts).area)
        h = building_height(area)
        cx = sum(q[0] for q in pts) / len(pts)
        cz = sum(q[1] for q in pts) / len(pts)
        FLOOR_TABLE = {(312, -339): 16, (388, -137): 14,
                      (115, -513): 9,   # building_001：R1图14号圈围合楼，主体标9F（实测2026-11-21）
                      (-88, -504): 8,   # building_003：R1图12号圈U形楼，主体标8F（实测2026-11-21）
                      (25, -183): 6,    # building_025a：教学区顶部复合楼，主楼标6F（实测2026-11-22）
                      (48, -130): 6,    # building_025b：教学区中部复合楼，主楼标6F（实测2026-11-22）
                      (74, -172): 2,    # building_025c：教学区右上配楼，标2F（实测2026-11-22）
                      (52, -81): 2}     # building_025d：教学区圆形楼，标2F（实测2026-11-22）
        # 规划图实测层数；增减楼层改此表
        # 宿舍区32栋（规划图14号圈，全部6F）—— 2026-11-21实证认定
        FLOOR_TABLE.update({(172, -356): 6, (208, -333): 6, (111, -324): 6, (127, -292): 6, (203, -300): 6, (113, -355): 6, (151, -386): 6, (232, -168): 6, (289, -140): 6, (283, -167): 6, (236, -106): 6, (199, -138): 6, (291, -108): 6, (372, -137): 6, (379, -107): 6, (372, -164): 6, (264, 5): 6, (209, -1): 6, (280, -27): 6, (272, 36): 6, (215, -33): 6, (192, 33): 6, (160, 109): 6, (90, 130): 6, (167, 141): 6, (88, 106): 6, (162, 82): 6, (93, 164): 6, (164, 176): 6, (96, 197): 6, (76, 70): 6, (283, 72): 6})
        fk = (round(cx), round(cz))
        if fk in FLOOR_TABLE:
            h = FLOOR_TABLE[fk] * 3.2 + 1.2
        roof_key = "roof"
        if cx > 100 and area >= 800:
            roof_key = "roof_blue"  # 东侧宿舍板楼：蓝灰顶
        elif area < 700 and ((int(cx * 13 + cz * 7)) % 3 + 3) % 3 == 0:
            roof_key = "roof_warm"  # 小体量建筑点缀橙黄顶
        body = TMesh()
        for t in tris:
            body.tri([(a, h, b) for a, b in t], (0, 1, 0),
                     [uv_top(a, b) for a, b in t])
        wall_key = "tower_wall" if h >= 25 else "building"
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            length = max(math.dist(a, b), 1.0)
            uv = [(0, 0), (length / 6, 0), (length / 6, h / 3.2), (0, h / 3.2)]
            body.quad([(a[0], 0, a[1]), (b[0], 0, b[1]),
                       (b[0], h, b[1]), (a[0], h, a[1])],
                      (a[1] - b[1], 0, b[0] - a[0]), uv)
        parts.append((body, wall_key))
        roof = TMesh()
        try:
            inner = Polygon(pts).buffer(-2.6, join_style=2)
            if not inner.is_empty and inner.geom_type == "Polygon":
                ip = list(inner.exterior.coords)[:-1]
                if len(ip) >= 3:
                    rise = min(4.0, max(3.0, h * 0.18))
                    for t in triangulate(Polygon(ip)):
                        if Polygon(ip).contains(t.representative_point()):
                            c = list(t.exterior.coords)[:3]
                            roof.tri([(a, h + rise, b) for a, b in c], (0, 1, 0),
                                     [uv_top(a, b) for a, b in c])
                    n = min(len(pts), len(ip))
                    for i in range(n):
                        a, b = pts[i], pts[(i + 1) % len(pts)]
                        c, d = ip[i], ip[(i + 1) % len(ip)]
                        roof.quad([(a[0], h, a[1]), (b[0], h, b[1]),
                                   (d[0], h + rise, d[1]), (c[0], h + rise, c[1])],
                                  (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        except Exception:
            pass
        if roof.pos:
            parts.append((roof, roof_key))
        for mm, mk in maotou_geometry(pts, h):
            parts.append((mm, mk))
    # 每栋楼：大门（最长边中点）+ 入口台阶
    for p in polys:
        if p["class"] != "building":
            continue
        pts = p["points"]
        n = len(pts)
        if n < 3:
            continue
        best, bA, bB = 0.0, pts[0], pts[1]
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            d2 = math.hypot(b[0] - a[0], b[1] - a[1])
            if d2 > best:
                best, bA, bB = d2, a, b
        if best < 6:
            continue
        ux, uz = (bB[0] - bA[0]) / best, (bB[1] - bA[1]) / best
        mx, mz = (bA[0] + bB[0]) / 2, (bA[1] + bB[1]) / 2
        ox, oz = uz, -ux
        ccx, ccz = sum(q[0] for q in pts) / n, sum(q[1] for q in pts) / n
        if (mx + ox * 3 - ccx) ** 2 + (mz + oz * 3 - ccz) ** 2 < (mx - ccx) ** 2 + (mz - ccz) ** 2:
            ox, oz = -ox, -oz
        dw, dh = 2.4, 2.8
        m = TMesh()
        m.quad([(mx - ux * dw / 2 + ox * 0.12, 0, mz - uz * dw / 2 + oz * 0.12),
                (mx + ux * dw / 2 + ox * 0.12, 0, mz + uz * dw / 2 + oz * 0.12),
                (mx + ux * dw / 2 + ox * 0.12, dh, mz + uz * dw / 2 + oz * 0.12),
                (mx - ux * dw / 2 + ox * 0.12, dh, mz - uz * dw / 2 + oz * 0.12)],
               (-uz, 0, ux), [(0, 0), (1, 0), (1, 1), (0, 1)])
        parts.append((m, "solid_3c4a63"))
        st1 = TMesh()
        st1.quad([(mx - ux * (dw / 2 + 1.0) + ox * 1.6, 0, mz - uz * (dw / 2 + 1.0) + oz * 1.6),
                  (mx + ux * (dw / 2 + 1.0) + ox * 1.6, 0, mz + uz * (dw / 2 + 1.0) + oz * 1.6),
                  (mx + ux * (dw / 2 + 1.0) + ox * 1.6, 0.36, mz + uz * (dw / 2 + 1.0) + oz * 1.6),
                  (mx - ux * (dw / 2 + 1.0) + ox * 1.6, 0.36, mz - uz * (dw / 2 + 1.0) + oz * 1.6)],
                 (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        st2 = TMesh()
        st2.quad([(mx - ux * (dw / 2 + 0.4) + ox * 0.7, 0, mz - uz * (dw / 2 + 0.4) + oz * 0.7),
                  (mx + ux * (dw / 2 + 0.4) + ox * 0.7, 0, mz + uz * (dw / 2 + 0.4) + oz * 0.7),
                  (mx + ux * (dw / 2 + 0.4) + ox * 0.7, 0.18, mz + uz * (dw / 2 + 0.4) + oz * 0.7),
                  (mx - ux * (dw / 2 + 0.4) + ox * 0.7, 0.18, mz - uz * (dw / 2 + 0.4) + oz * 0.7)],
                 (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        parts.append((st1, "solid_cfc9bb"))
        parts.append((st2, "solid_d9dde0"))
    for item in (details or []):
        parts.extend(detail_geometry(item))
    # 立体操场：北/南（绿茵场实测 bbox，跑道环绕外圈）
    parts.extend(stadium_geometry(274, -478.5, 65, 183, 47, 147))
    parts.extend(stadium_geometry(69, 427, 83.5, 204, 65.5, 168))
    # 北操场西看台（编号18）：跑道西缘向西逐级升高
    parts.extend(stand_geometry(207, -478, 70, 4, 3.0, 1.2))  # 仅北操场（图例18；南操场沙盘实拍无看台）
    parts.extend(rooftop_unit(-93, 317, 5.0, 2.5, 18.0))
    parts.extend(gate_geometry(-400, 0, 22.0, 9.0))
    return parts


def rooftop_unit(cx, cz, r=5.0, h=2.5, base_h=18.0):
    """楼顶圆形设备（风雨操场白色平顶上的圆筒）。"""
    seg = 14
    m = TMesh()
    for i in range(seg):
        a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
        p0 = (cx + r * math.cos(a0), cz + r * math.sin(a0))
        p1 = (cx + r * math.cos(a1), cz + r * math.sin(a1))
        m.quad([(p0[0], base_h, p0[1]), (p1[0], base_h, p1[1]),
                (p1[0], base_h + h, p1[1]), (p0[0], base_h + h, p0[1])],
               (1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        m.quad([(cx, base_h + h, cz), (p0[0], base_h + h, p0[1]), (p1[0], base_h + h, p1[1])],
               (0, 1, 0), [(.5, .5), (0, 0), (1, 0)])
    return [(m, "solid_d8d3c6")]


def gate_geometry(cx, cz, span=22.0, ph=9.0):
    """牌楼：双柱 + 横梁（跨路门楼）。"""
    parts = []
    for dz in (-span / 2 + 1.5, span / 2 - 1.5):
        m = TMesh()
        for (p, q) in (((cx - 1.5, cz + dz - 1.5), (cx + 1.5, cz + dz - 1.5)),
                       ((cx + 1.5, cz + dz - 1.5), (cx + 1.5, cz + dz + 1.5)),
                       ((cx + 1.5, cz + dz + 1.5), (cx - 1.5, cz + dz + 1.5)),
                       ((cx - 1.5, cz + dz + 1.5), (cx - 1.5, cz + dz - 1.5))):
            m.quad([(p[0], 0, p[1]), (q[0], 0, q[1]),
                    (q[0], ph, q[1]), (p[0], ph, p[1])],
                   (q[1] - p[1], 0, p[0] - q[0]), [(0, 0), (1, 0), (1, 1), (0, 1)])
        m.quad([(cx - 1.5, ph, cz + dz - 1.5), (cx + 1.5, ph, cz + dz - 1.5),
                (cx + 1.5, ph, cz + dz + 1.5), (cx - 1.5, ph, cz + dz + 1.5)],
               (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        parts.append((m, "solid_d9dde0"))
    beam = TMesh()
    beam.quad([(cx - span / 2 - 1, ph, cz - 1.5), (cx + span / 2 + 1, ph, cz - 1.5),
               (cx + span / 2 + 1, ph + 1.8, cz - 1.5), (cx - span / 2 - 1, ph + 1.8, cz - 1.5)],
              (0, 0, -1), [(0, 0), (1, 0), (1, 1), (0, 1)])
    beam.quad([(cx + span / 2 + 1, ph, cz + 1.5), (cx - span / 2 - 1, ph, cz + 1.5),
               (cx - span / 2 - 1, ph + 1.8, cz + 1.5), (cx + span / 2 + 1, ph + 1.8, cz + 1.5)],
              (0, 0, 1), [(0, 0), (1, 0), (1, 1), (0, 1)])
    beam.quad([(cx - span / 2 - 1, ph + 1.8, cz - 1.5), (cx + span / 2 + 1, ph + 1.8, cz - 1.5),
               (cx + span / 2 + 1, ph + 1.8, cz + 1.5), (cx - span / 2 - 1, ph + 1.8, cz + 1.5)],
              (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    beam.quad([(cx - span / 2 - 1, ph, cz - 1.5), (cx - span / 2 - 1, ph, cz + 1.5),
               (cx - span / 2 - 1, ph + 1.8, cz + 1.5), (cx - span / 2 - 1, ph + 1.8, cz - 1.5)],
              (-1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    beam.quad([(cx + span / 2 + 1, ph, cz - 1.5), (cx + span / 2 + 1, ph, cz + 1.5),
               (cx + span / 2 + 1, ph + 1.8, cz + 1.5), (cx + span / 2 + 1, ph + 1.8, cz - 1.5)],
              (1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    parts.append((beam, "solid_cfc9bb"))
    return parts


def stand_geometry(cx_edge, cz_center, length, tiers=4, tier_w=3.0, base_h=1.2):
    """阶梯看台：从场地边缘向西逐级升高（面向东侧场地）。"""
    parts = []
    for i in range(tiers):
        x1 = cx_edge - tier_w * i
        x0 = x1 - tier_w
        h = base_h * (i + 1)
        m = TMesh()
        m.quad([(x0, h, cz_center - length / 2), (x1, h, cz_center - length / 2),
                (x1, h, cz_center + length / 2), (x0, h, cz_center + length / 2)],
               (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        m.quad([(x1, 0, cz_center - length / 2), (x1, 0, cz_center + length / 2),
                (x1, h, cz_center + length / 2), (x1, h, cz_center - length / 2)],
               (-1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        parts.append((m, "solid_d9dde0" if i % 2 == 0 else "solid_cfc9bb"))
    return parts


def track_pts(cx, cz, ax, bs, n=7):
    """标准田径场边界点：两条直道（长 2*bs）+ 两端半圆（半径 ax）。"""
    pts = []
    for k in range(n + 1):
        pts.append((cx + ax, cz - bs + 2 * bs * k / n))          # 右直道（下→上）
    a = -math.pi / 2
    for k in range(n):                                            # 上端半圆
        a2 = -math.pi / 2 + math.pi * (k + 1) / n
        a0m = -math.pi / 2 + math.pi * k / n
        pts.append((cx + ax * math.cos((a0m + a2) / 2), cz + bs + ax * math.sin((a0m + a2) / 2)))
    for k in range(n + 1):
        pts.append((cx - ax, cz + bs - 2 * bs * k / n))          # 左直道（上→下）
    for k in range(n):
        a0m = math.pi / 2 + math.pi * k / n
        a2 = math.pi / 2 + math.pi * (k + 1) / n
        pts.append((cx + ax * math.cos((a0m + a2) / 2), cz - bs + ax * math.sin((a0m + a2) / 2)))
    return pts


def stadium_geometry(cx, cz, ax, bz, axi, bzi, seg=28):
    """立体操场（标准田径场形）：红色跑道环（h=0.4）+ 内圈绿茵场（h=0.18）。
    ax/bz=外圈半宽/总长，axi/bzi=内圈半宽/总长。"""
    H1, H2 = 0.4, 0.18
    ring, disc = TMesh(), TMesh()
    bs, bsi = (bz - 2 * ax) / 2, (bzi - 2 * axi) / 2
    O = track_pts(cx, cz, ax, bs)
    I = track_pts(cx, cz, axi, bsi)
    nseg = len(O)
    for i in range(nseg):
        o0, o1 = O[i], O[(i + 1) % nseg]
        i0, i1 = I[i % len(I)], I[(i + 1) % len(I)]
        _ = (a0, a1) = (0, 0)
        ring.quad([(o0[0], H1, o0[1]), (o1[0], H1, o1[1]),
                   (i1[0], H1, i1[1]), (i0[0], H1, i0[1])],
                  (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        ring.quad([(o0[0], 0, o0[1]), (o1[0], 0, o1[1]),
                   (o1[0], H1, o1[1]), (o0[0], H1, o0[1])],
                  (1, 0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    for i in range(len(I)):
        a, b = I[i], I[(i + 1) % len(I)]
        disc.tri([(cx, H2, cz), (a[0], H2, a[1]), (b[0], H2, b[1])],
                 (0, 1, 0), [(.5, .5), (0, 0), (1, 0)])
    return [(ring, "solid_c96a5c"), (disc, "solid_5f9e46")]


def long_edges(pts):
    """返回与最长边方向接近的边 [(A,B), ...] —— 即建筑两条长边（檐口矮墙沿其布置）。"""
    if len(pts) < 3:
        return []
    best, bA, bB = 0.0, pts[0], pts[1]
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        d = math.dist(a, b)
        if d > best:
            best, bA, bB = d, a, b
    if best < 1e-6:
        return []
    ang0 = math.atan2(bB[1] - bA[1], bB[0] - bA[0])
    edges = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        if math.dist(a, b) < 1.0:
            continue
        da = math.atan2(b[1] - a[1], b[0] - a[0]) - ang0
        da = abs(math.atan2(math.sin(da), math.cos(da)))
        if da < 0.35:  # ~20° 容差
            edges.append((a, b))
    return edges


def obox(cx, cz, ux, uz, vx, vz, hu, hv, base, top):
    """以 (ux,uz) 为长轴、(vx,vz) 为短轴的朝向盒子，用于马头墙阶梯。"""
    m = TMesh()
    def corner(su, sv, y):
        return (cx + su * hu * ux + sv * hv * vx,
                y,
                cz + su * hu * uz + sv * hv * vz)
    b = [corner(-1, -1, base), corner(1, -1, base), corner(1, 1, base), corner(-1, 1, base)]
    t = [corner(-1, -1, top), corner(1, -1, top), corner(1, 1, top), corner(-1, 1, top)]
    nrm = [(-ux, -uz), (vx, vz), (ux, uz), (-vx, -vz)]  # 四边外法线
    for i in range(4):
        a, c = i, (i + 1) % 4
        n = nrm[i]
        m.quad([b[a], b[c], t[c], t[a]], (n[0], 0, n[1]), [(0, 0), (1, 0), (1, 1), (0, 1)])
    m.quad([t[0], t[1], t[2], t[3]], (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
    return m


def maotou_along(A, B, h, n=4, thick=0.6, levels=None):
    """沿边 A->B 在檐口(y=h)上方生成矮墙带（黛瓦深灰），levels 可指定阶梯高度。"""
    if levels is None:
        levels = [0.5, 0.5, 0.5, 0.5]
    m = TMesh()
    dx, dy = B[0] - A[0], B[1] - A[1]
    L = math.hypot(dx, dy)
    if L < 2:
        return m
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    seg = L / n
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        ax, az = A[0] + dx * t0, A[1] + dy * t0
        bx, bz = A[0] + dx * t1, A[1] + dy * t1
        lvl = levels[i] if i < len(levels) else 1.0
        top = h + lvl
        a0 = (ax + nx * thick / 2, az + ny * thick / 2)
        a1 = (ax - nx * thick / 2, az - ny * thick / 2)
        b0 = (bx + nx * thick / 2, bz + ny * thick / 2)
        b1 = (bx - nx * thick / 2, bz - ny * thick / 2)
        m.quad([(a0[0], h, a0[1]), (b0[0], h, b0[1]), (b0[0], top, b0[1]), (a0[0], top, a0[1])],
               (nx, 0, ny), [(0, 0), (seg / thick, 0), (seg / thick, 1), (0, 1)])
        m.quad([(a1[0], h, a1[1]), (b1[0], h, b1[1]), (b1[0], top, b1[1]), (a1[0], top, a1[1])],
               (-nx, 0, -ny), [(0, 0), (seg / thick, 0), (seg / thick, 1), (0, 1)])
        m.quad([(a0[0], top, a0[1]), (b0[0], top, b0[1]), (b1[0], top, b1[1]), (a1[0], top, a1[1])],
               (0, 1, 0), [(0, 0), (seg / thick, 0), (seg / thick, 1), (0, 1)])
    return m


def point_in_poly(px, pz, pts):
    """射线法：点是否在轮廓多边形内。"""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, zi = pts[i]
        xj, zj = pts[j]
        if (zi > pz) != (zj > pz):
            xcross = (xj - xi) * (pz - zi) / (zj - zi + 1e-12) + xi
            if px < xcross:
                inside = not inside
        j = i
    return inside


def maotou_geometry(pts, h):
    """徽派马头墙：① 檐口黛瓦压顶（沿每条边一圈，对任意形状鲁棒、绝不悬空）；
    ② 长轴两端阶梯山墙——先探测端面实际覆盖区段再铺设，逐盒包含复查，
    切角/斜边/不规则形状的越界盒一律剔除。"""
    out = []
    n = len(pts)
    if n < 3:
        return out
    # ① 檐口压顶：每条边一圈矮墙（黛瓦）
    for i in range(n):
        A = pts[i]; B = pts[(i + 1) % n]
        m = maotou_along(A, B, h, levels=(1.0, 1.0, 1.0, 1.0), thick=0.6)
        if m.pos:
            out.append((m, "maotou"))
    # 长轴方向 = 最长边方向；OBB 极值以顶点均值为参考原点
    cx = sum(q[0] for q in pts) / n
    cz = sum(q[1] for q in pts) / n
    best, bA, bB = 0.0, pts[0], pts[1]
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dd = math.dist(a, b)
        if dd > best:
            best, bA, bB = dd, a, b
    if best < 1e-6:
        return out
    ux, uz = (bB[0] - bA[0]) / best, (bB[1] - bA[1]) / best
    vx, vz = -uz, ux
    us = [(q[0] - cx) * ux + (q[1] - cz) * uz for q in pts]
    vs = [(q[0] - cx) * vx + (q[1] - cz) * vz for q in pts]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    # ② 长轴两端阶梯山墙（马头墙）：先沿端面内缩线密采样找出楼面实际覆盖的
    #    最长连续区段，山墙只铺在该区段内；每盒再做包含复查，任何形状都安全。
    M = 5
    base, step = 1.2, 1.6
    thick = 1.4
    SAMP = 80
    for u_end, e in ((umax, 1), (umin, -1)):
        ui = u_end - e * thick * 0.5
        flags = [point_in_poly(cx + ui * ux + (vmin + (vmax - vmin) * s / SAMP) * vx,
                               cz + ui * uz + (vmin + (vmax - vmin) * s / SAMP) * vz, pts)
                 for s in range(SAMP + 1)]
        best_s = best_len = cur_s = cur_len = 0
        for s, f in enumerate(flags + [False]):
            if f:
                if cur_len == 0:
                    cur_s = s
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_s, best_len = cur_s, cur_len
                cur_len = 0
        if best_len < 4 or best_len / SAMP * (vmax - vmin) < 6.0:
            continue  # 端面过窄或缺失：该端不设山墙
        v_lo = vmin + (vmax - vmin) * best_s / SAMP
        v_hi = vmin + (vmax - vmin) * (best_s + best_len) / SAMP
        vseg = (v_hi - v_lo) / M
        for i in range(M):
            vc = v_lo + (i + 0.5) * vseg
            lvl = base + step * (2 - abs(i - 2))  # 0,1,2,1,0 → 中脊封顶
            ok = True
            for t5 in range(5):
                vv = (vc - vseg * 0.49) + (vseg * 0.98) * t5 / 4
                if not point_in_poly(cx + ui * ux + vv * vx, cz + ui * uz + vv * vz, pts):
                    ok = False
                    break
            if not ok:
                continue  # 区段斜边界处的角落盒：跳过，防悬空
            ccx = cx + u_end * ux + vc * vx
            ccz = cz + u_end * uz + vc * vz
            m = obox(ccx, ccz, ux, uz, vx, vz, thick * 0.5, vseg * 0.49, h, h + lvl)
            if m.pos:
                out.append((m, "maotou"))
    return out


def build_glb(parts, ground, textures=None):
    bin_data = bytearray()
    bvs, accs, prims, mats, imgs = [], [], [], [], []
    tri_count = 0

    def add_view(data, target=None):
        while len(bin_data) % 4:
            bin_data.append(0)
        off = len(bin_data)
        bin_data.extend(data)
        while len(bin_data) % 4:
            bin_data.append(0)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        bvs.append(v)
        return len(bvs) - 1

    # 地面：把官方规划图按像素→世界映射贴到 y=0 平面
    if ground:
        gm = TMesh()
        X0, Z0, X1, Z1 = -497.24, -661.05, 497.24, 661.05
        gm.quad([(X0, 0, Z0), (X1, 0, Z0), (X1, 0, Z1), (X0, 0, Z1)],
                (0, 1, 0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        parts = [(gm, "ground")] + parts

    tex_objs = []
    ground_tex_index = None
    if ground:
        imgs.append({"bufferView": add_view(ground),
                     "mimeType": "image/jpeg", "name": "ground"})
        ground_tex_index = len(tex_objs)
        tex_objs.append({"sampler": 0, "source": len(imgs) - 1})
    tex_index_by_name = {}
    if textures:
        for tname in ("wall", "roof", "roof_blue", "roof_warm", "tower_wall"):
            if textures.get(tname):
                imgs.append({"bufferView": add_view(textures[tname]),
                             "mimeType": "image/png", "name": tname})
                ti = len(tex_objs)
                tex_objs.append({"sampler": 0, "source": len(imgs) - 1})
                tex_index_by_name[tname] = ti
    wall_tex_index = tex_index_by_name.get("wall")
    roof_tex_index = tex_index_by_name.get("roof")
    mat_index = {}

    def get_mat(key):
        if key in mat_index:
            return mat_index[key]
        idx = len(mats)
        mat_index[key] = idx
        if key == "ground":
            mats.append({"name": "ground", "doubleSided": True,
                         "pbrMetallicRoughness": {
                             "baseColorTexture": {"index": ground_tex_index},
                             "baseColorFactor": [1, 1, 1, 1],
                             "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        elif key == "maotou":
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorFactor": [38/255, 43/255, 48/255, 1.0],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.85}})
        elif key == "building" and wall_tex_index is not None:
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorTexture": {"index": wall_tex_index},
                                                  "baseColorFactor": [1, 1, 1, 1],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.95}})
        elif key == "building" and tex_index_by_name.get("tower_wall") is not None and False:
            pass  # 占位（building 材质保持在下方分支）
        elif key == "tower_wall" and tex_index_by_name.get("tower_wall") is not None:
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorTexture": {"index": tex_index_by_name["tower_wall"]},
                                                  "baseColorFactor": [1, 1, 1, 1],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.9}})
        elif key in ("roof", "roof_blue", "roof_warm") and tex_index_by_name.get(key) is not None:
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorTexture": {"index": tex_index_by_name[key]},
                                                  "baseColorFactor": [1, 1, 1, 1],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.8}})
        elif key.startswith("solid_"):
            hx = key[6:]
            rgb = [int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorFactor": rgb + [1.0],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.9}})
        else:
            col = FLAT.get(key, (200, 200, 200))
            rgb = [c / 255.0 for c in col]
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {
                             "baseColorFactor": rgb + [1.0],
                             "metallicFactor": 0.05 if key == "water" else 0.0,
                             "roughnessFactor": 0.5 if key == "water" else 0.9}})
        return idx

    for mesh, key in parts:
        if not mesh.pos:
            continue
        mi = get_mat(key)
        fp = [c for p in mesh.pos for c in p]
        fn = [c for n in mesh.nrm for c in n]
        fu = [c for u in mesh.uv for c in u]
        pv = add_view(struct.pack("<%df" % len(fp), *fp), 34962)
        nv = add_view(struct.pack("<%df" % len(fn), *fn), 34962)
        tv = add_view(struct.pack("<%df" % len(fu), *fu), 34962)
        iv = add_view(struct.pack("<%dH" % len(mesh.idx), *mesh.idx), 34963)
        pa = len(accs)
        accs.append({"bufferView": pv, "componentType": 5126, "count": len(mesh.pos), "type": "VEC3",
                     "min": [min(p[i] for p in mesh.pos) for i in range(3)],
                     "max": [max(p[i] for p in mesh.pos) for i in range(3)]})
        na = len(accs)
        accs.append({"bufferView": nv, "componentType": 5126, "count": len(mesh.nrm), "type": "VEC3"})
        ta = len(accs)
        accs.append({"bufferView": tv, "componentType": 5126, "count": len(mesh.uv), "type": "VEC2"})
        ia = len(accs)
        accs.append({"bufferView": iv, "componentType": 5123, "count": len(mesh.idx), "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": pa, "NORMAL": na, "TEXCOORD_0": ta},
                      "indices": ia, "material": mi, "mode": 4})
        tri_count += len(mesh.idx) // 3

    gltf = {"asset": {"version": "2.0", "generator": "campus-groundmap-v1"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "方山校区 1:1 轮廓 + 真实规划图地面"}],
            "meshes": [{"name": "campus_textured", "primitives": prims}],
            "materials": mats, "textures": tex_objs,
            "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
            "images": imgs,
            "buffers": [{"byteLength": len(bin_data)}],
            "bufferViews": bvs, "accessors": accs}
    j = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * ((4 - len(j) % 4) % 4)
    b = bytes(bin_data)
    total = 12 + 8 + len(j) + 8 + len(b)
    glb = (struct.pack("<4sII", b"glTF", 2, total)
           + struct.pack("<I4s", len(j), b"JSON") + j
           + struct.pack("<I4s", len(b), b"BIN\x00") + b)
    open(GLB, "wb").write(glb)
    return len(glb), tri_count, len(prims)


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def make_preview(polys, details, labels, ground, trees=None):
    data = [{"c": p["class"], "p": p["points"]} for p in polys]
    cmap = json.load(open(os.path.join(ROOT, "data", "_build_color_map.json"), encoding="utf-8"))
    det = [{"t": d["t"], "x": d["x"], "z": d["z"], "w": d["w"],
            "d": d["d"], "h": d["h"], "c": d["c"], "n": d["n"]} for d in details]
    ground_b64 = base64.b64encode(ground).decode() if ground else ""
    imgs = {"plan": ground_b64,
            "overlay": b64(os.path.join(ROOT, "docs", "outline_overlay.jpg")),
            "ref": b64(os.path.join(ROOT, "refs", "official_02_birdview.jpg"))}
    html = r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>方山校区 · 1:1 轮廓 + 材质光影</title>
<style>
body{margin:0;padding:22px 28px 50px;background:#fff;color:#1f2328;font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1240px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}h2{font-size:16px;font-weight:600;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #d8dee4}
h3{font-size:14px;font-weight:600;margin:0 0 8px}.sub,.note{color:#57606a;font-size:12.5px}
.status{background:#f6f8fa;border:1px solid #d8dee4;border-left:3px solid #1f6feb;padding:10px 14px;margin:14px 0 18px;font-size:13px}
.split{display:flex;gap:16px;flex-wrap:wrap}.pane{flex:1 1 360px;min-width:290px;border:1px solid #d8dee4;padding:10px}
.pane img{width:100%;display:block;border:1px solid #eaeef2}
canvas{width:100%;height:580px;background:#eef3f7;border:1px solid #eaeef2;display:block;cursor:grab}
.ctl{font-size:12.5px;color:#57606a;margin-top:8px}button{border:1px solid #d8dee4;background:#f6f8fa;padding:3px 10px;cursor:pointer}
.note{background:#f6f8fa;border:1px solid #eaeef2;padding:10px 14px;margin:12px 0}
</style></head><body>
<h1>南京晓庄学院（方山校区）· 1:1 轮廓 + 真实规划图地面</h1>
<p class="sub">几何形状逐像素取自官方总平面图，坐标由图上 100m 比例尺定标；地面直接铺真实规划图（与轮廓同一坐标系，俯视即为校园地图），并标注主要功能区。建筑已按徽派风格（青砖小瓦马头墙）程序化上色并出挑山墙。</p>
<div class="status">本页<b>自包含、不联网</b>。左下角「规划总平面图」与三维场景地面是同一张图。<br>建筑按徽派（青砖小瓦马头墙）风格程序化上色与出挑山墙；坐标为线性映射示意值，非测绘成果。</div>
<div class="split">
<div class="pane"><h3>官方规划总平面图（地面底图）</h3><img id="planimg" src="data:image/jpeg;base64,__PLAN__" alt="总平面图"></div>
<div class="pane"><h3>云端描边验收</h3><img src="data:image/jpeg;base64,__OVERLAY__" alt="描边验收"></div>
<div class="pane"><h3>对标效果图（徽派写实渲染）</h3><img src="data:image/jpeg;base64,__REF__" alt="对标"></div>
</div>
<h2>校园地图与三维预览</h2>
<div class="ctl"><button onclick="setView(2)">俯视地图</button> <button onclick="setView(3)">三维预览</button> <button onclick="setView(4)">R2效果视角</button>
<span id="hint">默认俯视图：真实规划图 + 楼栋填色 + 中文标注 + 100m 比例尺</span></div>
<canvas id="cv" width="1400" height="920" style="width:100%;max-width:1400px"></canvas>
<div class="ctl" id="ctl3d" style="display:none">左右拖动旋转，上下平移，滚轮缩放　<button onclick="resetView()">重置视角</button>
<label><input type="checkbox" id="ground" checked onchange="draw()"> 规划图地面</label>
<label><input type="checkbox" id="label" checked onchange="draw()"> 标注</label>
<label><input type="checkbox" id="tree" checked onchange="draw()"> 树木</label>
<button id="walkbtn" onclick="toggleWalk()">🚶 进入漫游</button></div>
<div class="note">共 __COUNT__ 个真实轮廓。来源：学校发展规划处公开总平面规划图；实拍来自公开媒体转载，版权归原发布方。教学展示用途 · 示意性布局 · 非测绘成果。</div>
<script>
const DATA=__DATA__;
const BCOL=__BCOL__; // R1规划图逐像素实测楼体颜色（data/building_colors.json）
const DETAILS=__DETAILS__;
const LABELS=__LABELS__;
const GROUND=document.getElementById('planimg').src.split(',')[1];
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const GIMG=new Image();GIMG.onload=draw;GIMG.src='data:image/jpeg;base64,'+GROUND;
const BASE={building:'#e0e2e1',roof:'#6c7278',maotou:'#22272c',water:'#6fa8cf',green:'#7fa85a',road:'#c0c4bc'};
const TREES=__TREES__;
const TGREEN=['#5d8a4a','#6a975a','#4f7d42','#77a163'];
let ang=-.62,zoom=1,panX=0,panY=140,drag=false,lx=0,ly=0,view=2;
function baseScale(){return Math.min(cv.width/2100,cv.height/1120);}
function sh(h,f){const s=h.replace('#','');return 'rgb('+[0,2,4].map(i=>Math.min(255,parseInt(s.slice(i,i+2),16)*f|0)).join(',')+')';}
function proj(x,y,z){const c=Math.cos(ang),s=Math.sin(ang),rx=x*c-z*s,rz=x*s+z*c;
return [cv.width/2+(rx-rz)*.866*sc+panX,cv.height/2+(rx+rz)*.5*sc-y*sc*1.45+panY];}
let walkMode=false,camX=-380,camZ=0,camH=1.6,camYaw=0,camPitch=0,mapZoom=false,walkKeys={},walkFaces=[],lastWT=0,WCOL=[];
const FLORTAB={'312,-339':16,'388,-137':14,
  '115,-513':9,   // building_001：R1图14号圈围合楼主体9F（实测2026-11-21）
  '-88,-504':8,   // building_003：R1图12号圈U形楼主体8F（实测2026-11-21）
  '25,-183':6,    // building_025a：教学区顶部复合楼，主楼标6F（实测2026-11-22）
  '48,-130':6,    // building_025b：教学区中部复合楼，主楼标6F（实测2026-11-22）
  '74,-172':2,    // building_025c：教学区右上配楼，标2F（实测2026-11-22）
  '52,-81':2};    // building_025d：教学区圆形楼，标2F（实测2026-11-22）
// 楼层表：键=楼栋质心取整，值=层数；增减楼层改这里
// 宿舍区32栋（规划图14号圈，全部6F）—— 2026-11-21实证认定
Object.assign(FLORTAB,{'172,-356':6, '208,-333':6, '111,-324':6, '127,-292':6, '203,-300':6, '113,-355':6, '151,-386':6, '232,-168':6, '289,-140':6, '283,-167':6, '236,-106':6, '199,-138':6, '291,-108':6, '372,-137':6, '379,-107':6, '372,-164':6, '264,5':6, '209,-1':6, '280,-27':6, '272,36':6, '215,-33':6, '192,33':6, '160,109':6, '90,130':6, '167,141':6, '88,106':6, '162,82':6, '93,164':6, '164,176':6, '96,197':6, '76,70':6, '283,72':6});
function FOC(){return cv.height*1.1;}
function camSpace(x,y,z){const dx=x-camX,dz=z-camZ,dy=y-camH,c=Math.cos(camYaw),s2=Math.sin(camYaw);
const right=-dx*s2+dz*c,fwd=dx*c+dz*s2,cp=Math.cos(camPitch),sp=Math.sin(camPitch);
return [right,fwd*cp+dy*sp,dy*cp-fwd*sp,x,y,z];}
function clipNear(pts){const NEAR=.06,out=[];for(let i=0;i<pts.length;i++){const a=pts[i],b=pts[(i+1)%pts.length];
const ain=a[1]>=NEAR,bin=b[1]>=NEAR;if(ain)out.push(a);
if(ain!==bin){const t=(NEAR-a[1])/(b[1]-a[1]);out.push([a[0]+(b[0]-a[0])*t,NEAR,a[2]+(b[2]-a[2])*t,
a[3]+(b[3]-a[3])*t,a[4]+(b[4]-a[4])*t,a[5]+(b[5]-a[5])*t]);}}return out;}
function wproj(q){const F=FOC();return [cv.width/2+(q[0]/q[1])*F,cv.height/2-(q[2]/q[1])*F];}
function wpoly(pts,fill,stroke,bias){const cs=pts.map(p=>camSpace(p[0],p[1],p[2]));const cl=clipNear(cs);
if(cl.length<3)return;const davg=cl.reduce((s2,q)=>s2+q[1],0)/cl.length-(bias||0);
walkFaces.push({P:cl.map(q=>wproj(q)),d:davg,col:fill,st:stroke||null});}
function camD(x,z){const dx=x-camX,dz=z-camZ;return dx*Math.cos(camYaw)+dz*Math.sin(camYaw);}
function ptin(x,z,pts){let ins=false;for(let i=0,j=pts.length-1;i<pts.length;j=i++){const xi=pts[i][0],zi=pts[i][1],xj=pts[j][0],zj=pts[j][1];
if(((zi>z)!==(zj>z))&&(x<(xj-xi)*(z-zi)/(zj-zi)+xi))ins=!ins;}return ins;}
function blockedH(nx,nz,ch){if(Math.abs(nx)>500||Math.abs(nz)>655)return true;
for(const p of WCOL){if(nx<p.bx0||nx>p.bx1||nz<p.bz0||nz>p.bz1)continue;
if(ch<p.h+0.6&&ptin(nx,nz,p.p))return true;}return false;}
function blocked(nx,nz){return blockedH(nx,nz,camH);}
function mouseXY(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];}
function toggleWalk(){walkMode=!walkMode;
if(walkMode){camX=-380;camZ=0;camYaw=0;camH=1.6;walkKeys={};
WCOL=DATA.filter(r=>r.c==='building'||r.c==='water').map(r=>{const xs=r.p.map(q=>q[0]),zs=r.p.map(q=>q[1]);
const a=area(r.p);let hh=r.c==='water'?0.3:height(a);const fk=Math.round(r.p.reduce((s2,q)=>s2+q[0],0)/r.p.length)+','+Math.round(r.p.reduce((s2,q)=>s2+q[1],0)/r.p.length);
if(FLORTAB[fk])hh=FLORTAB[fk]*3.2+1.2;
return{p:r.p,bx0:Math.min(...xs),bx1:Math.max(...xs),bz0:Math.min(...zs),bz1:Math.max(...zs),h:hh};});
document.getElementById('walkbtn').textContent='🚪 退出漫游';
lastWT=performance.now();draw();walkLoop();}
else{document.getElementById('walkbtn').textContent='🚶 进入漫游';draw();}}
function walkLoop(){if(!walkMode)return;
const now=performance.now();const dt=Math.min(.1,(now-lastWT)/1000);lastWT=now;
const mv=((walkKeys['w']||walkKeys['arrowup']?1:0)-(walkKeys['s']||walkKeys['arrowdown']?1:0))*(walkKeys['shift']?1.7:1);
const st=(walkKeys['d']?1:0)-(walkKeys['a']?1:0);
const tn=(walkKeys['q']||walkKeys['arrowleft']?1:0)-(walkKeys['e']||walkKeys['arrowright']?1:0);
const up=(walkKeys[' ']?1:0)-(walkKeys['c']||walkKeys['control']?1:0);
if(tn)camYaw+=tn*2.2*dt;
if(up)camH=Math.max(1.6,Math.min(80,camH+up*9*dt));
if(mv||st){const c=Math.cos(camYaw),s2=Math.sin(camYaw);
const vx=(c*mv-s2*st)*16*dt,vz=(s2*mv+c*st)*16*dt;
const nx=camX+vx,nz=camZ+vz;const hlim=blockedH(nx,nz,camH);
if(!hlim){camX=nx;camZ=nz;}else if(!blockedH(camX+vx,camZ,camH)){camX+=vx;}else if(!blockedH(nx,camZ+vz,camH)){camZ+=vz;}}
draw();requestAnimationFrame(walkLoop);}
function poly3(pts,fill,stroke,bias){if(walkMode){wpoly(pts,fill,stroke,bias);return;}ctx.beginPath();pts.forEach((p,i)=>{const q=proj(p[0],p[1],p[2]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke||'rgba(35,45,60,.25)';ctx.lineWidth=.5;ctx.stroke();}
function area(p){let a=0;for(let i=0;i<p.length;i++){const q=p[(i+1)%p.length];a+=p[i][0]*q[1]-q[0]*p[i][1];}return Math.abs(a)/2;}
function height(a){return a>9000?21:(a>3500?18:(a>1200?15:11));}
function drawDetail(d){
if(d.t==='treeline'){const n=Math.max(6,Math.floor(d.d/45));
if(walkMode){for(let i=0;i<n;i++){const zz=d.z-d.d/2+d.d*(i+.5)/n;
const cs2=camSpace(d.x,d.h*.62,zz);if(cs2[1]<.5)continue;
const pc=wproj(cs2);const rr=Math.max(1.2,3.2*FOC()/cs2[1]);
const pts=[];for(let k=0;k<6;k++){const a=Math.PI*2*k/6+1;pts.push([pc[0]+rr*Math.cos(a),pc[1]+rr*Math.sin(a)]);}
walkFaces.push({P:pts,d:cs2[1],col:d.c,st:null});
const qb=camSpace(d.x,0,zz);if(qb[1]<.4)continue;const pb=wproj(qb);const tw2=Math.max(1,.3*FOC()/qb[1]);
walkFaces.push({P:[[pb[0]-tw2,pb[1]],[pb[0]+tw2,pb[1]],[pc[0]+tw2*.5,pc[1]],[pc[0]-tw2*.5,pc[1]]],d:qb[1],col:'#8a7256',st:null});}
return;}
for(let i=0;i<n;i++){const zz=d.z-d.d/2+d.d*(i+.5)/n,q=proj(d.x,d.h*.62,zz);
ctx.beginPath();ctx.arc(q[0],q[1],4.6,0,7);ctx.fillStyle=d.c;ctx.fill();
ctx.beginPath();ctx.arc(q[0],q[1]+5,1.8,0,7);ctx.fillStyle='#8a7256';ctx.fill();}return;}
if(d.t==='track'){if(walkMode)return;const x0=d.x-d.w/2,x1=d.x+d.w/2,z0=d.z-d.d/2,z1=d.z+d.d/2;
ctx.beginPath();[[x0,.3,z0],[x1,.3,z0],[x1,.3,z1],[x0,.3,z1],[x0,.3,z0]].forEach((p,i)=>{const q=proj(p[0],p[1],p[2]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
ctx.strokeStyle=d.c;ctx.lineWidth=3;ctx.stroke();return;}
if(d.t==='statue'||d.t==='roundbed'||d.t==='pavilion'||d.t==='tower'){if(walkMode)return;
const r=Math.min(d.w,d.d)/2,q0=proj(d.x,0,d.z),q1=proj(d.x,d.h,d.z),w=r*sc;
ctx.beginPath();ctx.moveTo(q0[0]-w,q0[1]);ctx.lineTo(q0[0]+w,q0[1]);ctx.lineTo(q1[0]+w,q1[1]);ctx.lineTo(q1[0]-w,q1[1]);ctx.closePath();
ctx.fillStyle=d.c;ctx.fill();ctx.strokeStyle='rgba(40,50,60,.35)';ctx.lineWidth=.6;ctx.stroke();
if(d.t==='pavilion'||d.t==='tower'){const t=proj(d.x,d.h+d.h*.42,d.z);
ctx.beginPath();ctx.moveTo(q1[0]-w*1.5,q1[1]);ctx.lineTo(t[0],t[1]);ctx.lineTo(q1[0]+w*1.5,q1[1]);ctx.closePath();ctx.fillStyle='#3c4a63';ctx.fill();}return;}
const x0=d.x-d.w/2,x1=d.x+d.w/2,z0=d.z-d.d/2,z1=d.z+d.d/2;
poly3([[x0,0,z0],[x1,0,z0],[x1,d.h,z0],[x0,d.h,z0]],d.c);
poly3([[x0,d.h,z0],[x1,d.h,z0],[x1,d.h,z1],[x0,d.h,z1]],d.c);
}
function fillBg(){ctx.setTransform(1,0,0,1,0,0);const g=ctx.createLinearGradient(0,0,0,cv.height);g.addColorStop(0,'#e9ece6');g.addColorStop(.55,'#eef1ea');g.addColorStop(1,'#e3e6dc');ctx.fillStyle=g;ctx.fillRect(0,0,cv.width,cv.height);}
function groundMatrix(){const W=1800,H=2395,ppm=1.81;const wx=u=>(u-900)/ppm,wz=v=>(v-1197.5)/ppm;
const s00=proj(wx(0),0,wz(0)),s10=proj(wx(W),0,wz(0)),s01=proj(wx(0),0,wz(H));
return {a:(s10[0]-s00[0])/W,b:(s10[1]-s00[1])/W,c:(s01[0]-s00[0])/H,d:(s01[1]-s00[1])/H,e:s00[0],f:s00[1]};}
function drawBase(){const X0=-505,Z0=-668,X1=505,Z1=668;
const top=[[X0,0,Z0],[X1,0,Z0],[X1,0,Z1],[X0,0,Z1]].map(q=>proj(q[0],0,q[2]));
const front=[[X0,0,Z1],[X1,0,Z1],[X1,-0.6,Z1],[X0,-0.6,Z1]].map(q=>proj(q[0],q[1],q[2]));
const right=[[X1,0,Z1],[X1,0,Z0],[X1,-0.6,Z0],[X1,-0.6,Z1]].map(q=>proj(q[0],q[1],q[2]));
ctx.fillStyle='#c9c7bb';ctx.beginPath();top.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fill();
ctx.fillStyle='#a9a79b';ctx.beginPath();front.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fill();
ctx.fillStyle='#b6b4a8';ctx.beginPath();right.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fill();}
function drawGround(){if(!GIMG.complete||!GIMG.naturalWidth)return;const m=groundMatrix();
ctx.save();ctx.setTransform(m.a,m.b,m.c,m.d,m.e,m.f);ctx.drawImage(GIMG,0,0);ctx.restore();ctx.setTransform(1,0,0,1,0,0);}
function drawLabels(){if(walkMode)return;if(!document.getElementById('label').checked)return;
ctx.textAlign='center';const zoomed=zoom>1.6;
LABELS.forEach(L=>{
// 楼栋号（A-1式）只在放大后显示，避免鸟瞰时标签糊成一团
if(/^[A-E]-\d+$/.test(L.n)&&!zoomed)return;
const small=/^[A-E]-\d+$/.test(L.n);
ctx.font=(small?'700 11px':'600 13px')+' "Microsoft YaHei",sans-serif';
const q=proj(L.x,8,L.z);const tw=ctx.measureText(L.n).width;
ctx.fillStyle=small?'rgba(255,255,255,.92)':'rgba(255,255,255,.88)';ctx.fillRect(q[0]-tw/2-4,q[1]-14,tw+8,17);
ctx.strokeStyle=small?'rgba(224,60,49,.5)':'rgba(31,111,235,.65)';ctx.lineWidth=1;ctx.strokeRect(q[0]-tw/2-4,q[1]-14,tw+8,17);
ctx.fillStyle=small?'#c0392b':'#1f2328';ctx.fillText(L.n,q[0],q[1]-2);});}
function longEdges(p){
let best=0,bA=p[0],bB=p[1];
for(let i=0;i<p.length;i++){const a=p[i],b=p[(i+1)%p.length],d=Math.hypot(b[0]-a[0],b[1]-a[1]);if(d>best){best=d;bA=a;bB=b;}}
if(best<1e-6)return[];
const ang0=Math.atan2(bB[1]-bA[1],bB[0]-bA[0]),edges=[];
for(let i=0;i<p.length;i++){const a=p[i],b=p[(i+1)%p.length];if(Math.hypot(b[0]-a[0],b[1]-a[1])<1)continue;
const da=Math.atan2(b[1]-a[1],b[0]-a[0])-ang0;const daa=Math.abs(Math.atan2(Math.sin(da),Math.cos(da)));
if(daa<0.35)edges.push([a,b]);}
return edges;}
function maotouOBB(p){
let cx=0,cz=0;for(const q of p){cx+=q[0];cz+=q[1];}cx/=p.length;cz/=p.length;
let best=0,bA=p[0],bB=p[1];
for(let i=0;i<p.length;i++){const a=p[i],b=p[(i+1)%p.length],d=Math.hypot(b[0]-a[0],b[1]-a[1]);if(d>best){best=d;bA=a;bB=b;}}
const ux=(bB[0]-bA[0])/best,uz=(bB[1]-bA[1])/best,vx=-uz,vz=ux;
let Lh=0,Wh=0;
for(const q of p){const pu=(q[0]-cx)*ux+(q[1]-cz)*uz,pv=(q[0]-cx)*vx+(q[1]-cz)*vz;Lh=Math.max(Lh,Math.abs(pu));Wh=Math.max(Wh,Math.abs(pv));}
return {cx,cz,ux,uz,vx,vz,Lh,Wh};}
function drawMaotou(p,h){
const n=p.length;if(n<3)return;
if(walkMode){for(let i=0;i<n;i++){const A=p[i],B=p[(i+1)%n];const L=Math.hypot(B[0]-A[0],B[1]-A[1]);if(L<2)continue;
wpoly([[A[0],h,A[1]],[B[0],h,B[1]],[B[0],h+1.0,B[1]],[A[0],h+1.0,A[1]]],BASE.maotou);}return;}
// 檐口黛瓦压顶：每条边一圈矮墙
for(let i=0;i<n;i++){const A=p[i],B=p[(i+1)%n],dx=B[0]-A[0],dy=B[1]-A[1],L=Math.hypot(dx,dy);if(L<2)continue;
const bot=[],top=[];
for(let j=0;j<=4;j++){const t=j/4,x=A[0]+dx*t,z=A[1]+dy*t;bot.push(proj(x,h,z));top.push(proj(x,h+1.0,z));}
ctx.beginPath();bot.forEach((q,j)=>j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));
for(let j=4;j>=0;j--)ctx.lineTo(top[j][0],top[j][1]);
ctx.closePath();ctx.fillStyle=BASE.maotou;ctx.fill();}
// 长轴两端阶梯山墙（马头墙）：逐盒包含测试，越界盒剔除，任何形状都安全
let cx=0,cz=0;for(const q of p){cx+=q[0];cz+=q[1];}cx/=n;cz/=n;
let best=0,bA=p[0],bB=p[0];
for(let i=0;i<n;i++){const a=p[i],b=p[(i+1)%n],d=Math.hypot(b[0]-a[0],b[1]-a[1]);if(d>best){best=d;bA=a;bB=b;}}
if(best<1e-6)return;
const ux=(bB[0]-bA[0])/best,uz=(bB[1]-bA[1])/best,vx=-uz,vz=ux;
let umin=1e9,umax=-1e9,vmin=1e9,vmax=-1e9;
for(const q of p){const pu=(q[0]-cx)*ux+(q[1]-cz)*uz,pv=(q[0]-cx)*vx+(q[1]-cz)*vz;
if(pu<umin)umin=pu;if(pu>umax)umax=pu;if(pv<vmin)vmin=pv;if(pv>vmax)vmax=pv;}
function pip(x,z){let inside=false;for(let a=0,b=n-1;a<n;b=a++){const xa=p[a][0],za=p[a][1],xb=p[b][0],zb=p[b][1];
if((za>z)!==(zb>z)&&x<(xb-xa)*(z-za)/(zb-za+1e-12)+xa)inside=!inside;}return inside;}
const M=5,base=1.2,step=1.6,thick=1.4,SAMP=80;
for(const [u_end,e] of [[umax,1],[umin,-1]]){
const ui=u_end-e*thick*0.5;
const flags=[];for(let s2=0;s2<=SAMP;s2++){const vv=vmin+(vmax-vmin)*s2/SAMP;flags.push(pip(cx+ui*ux+vv*vx,cz+ui*uz+vv*vz));}
let bs=0,bl=0,rs=0,rl=0;
for(let s2=0;s2<=SAMP+1;s2++){const f=s2<=SAMP?flags[s2]:false;
if(f){if(rl===0)rs=s2;rl++;}
else{if(rl>bl){bs=rs;bl=rl;}rl=0;}}
if(bl<4||bl/SAMP*(vmax-vmin)<6)continue;
const v_lo=vmin+(vmax-vmin)*bs/SAMP,v_hi=vmin+(vmax-vmin)*(bs+bl)/SAMP,vseg=(v_hi-v_lo)/M;
for(let i=0;i<M;i++){const vc=v_lo+(i+0.5)*vseg,lvl=base+step*(2-Math.abs(i-2));
let ok=true;
for(let t5=0;t5<5;t5++){const vv=vc-vseg*0.49+(vseg*0.98)*t5/4;
if(!pip(cx+ui*ux+vv*vx,cz+ui*uz+vv*vz)){ok=false;break;}}
if(!ok)continue;
const ccx=cx+u_end*ux+vc*vx,ccz=cz+u_end*uz+vc*vz,hu=thick*0.5,hv=vseg*0.49;
const cs=[[-hu,-hv],[hu,-hv],[hu,hv],[-hu,hv]].map(s=>[ccx+s[0]*ux+s[1]*vx,ccz+s[0]*uz+s[1]*vz]);
const o=cs.map(c=>proj(c[0],h,c[1])),t=cs.map(c=>proj(c[0],h+lvl,c[1]));
ctx.beginPath();o.forEach((q,j)=>j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));
for(let k=3;k>=0;k--)ctx.lineTo(t[k][0],t[k][1]);
ctx.closePath();ctx.fillStyle=BASE.maotou;ctx.fill();
ctx.strokeStyle='rgba(18,22,26,.5)';ctx.lineWidth=.5;ctx.stroke();}}
}
function drawTree(t){const showT=document.getElementById('tree').checked;if(!showT)return;
if(walkMode){const d=camD(t.x,t.z);if(d<.8||d>420)return;
const gc=(Math.floor(Math.abs(t.x*7+t.z*3))%8===7)?'#d9a441':TGREEN[Math.floor(Math.abs(t.x*7+t.z*3))%4];
const q0=camSpace(t.x,1.0,t.z),qc=camSpace(t.x,t.h*.78,t.z);if(q0[1]<.4||qc[1]<.4)return;
const p0=wproj(q0),pc=wproj(qc);const rr=Math.max(1.2,t.r*FOC()/qc[1]);
const pts=[];for(let i=0;i<6;i++){const a=Math.PI*2*i/6+1;pts.push([pc[0]+rr*Math.cos(a),pc[1]+rr*Math.sin(a)]);}
walkFaces.push({P:pts,d:qc[1],col:gc,st:null});
const tw=Math.max(1,.28*FOC()/q0[1]);
walkFaces.push({P:[[p0[0]-tw,p0[1]],[p0[0]+tw,p0[1]],[pc[0]+tw*.5,pc[1]],[pc[0]-tw*.5,pc[1]]],d:q0[1],col:'#6b4f3a',st:null});
return;}
const q0=proj(t.x,0.4,t.z),q1=proj(t.x,t.h*.55,t.z),qc=proj(t.x,t.h*.78,t.z);
ctx.strokeStyle='#6b4f3a';ctx.lineWidth=Math.max(1,t.r*sc*.35);
ctx.beginPath();ctx.moveTo(q0[0],q0[1]);ctx.lineTo(q1[0],q1[1]);ctx.stroke();
ctx.beginPath();ctx.arc(qc[0],qc[1],Math.max(1.5,t.r*sc*.85),0,7);
const hs=Math.floor(Math.abs(t.x*7+t.z*3));ctx.fillStyle=(hs%8===7)?'#d9a441':TGREEN[hs%4];ctx.fill();
ctx.strokeStyle='rgba(40,62,34,.35)';ctx.lineWidth=.5;ctx.stroke();}
function box3(x0,y0,z0,x1,y1,z1,col){if(walkMode){
wpoly([[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],col);
wpoly([[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],col);
wpoly([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],col);
wpoly([[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],col);
wpoly([[x1,y0,z1],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],col);
wpoly([[x0,y0,z1],[x0,y0,z0],[x0,y1,z0],[x0,y1,z1]],col);return;}
const P=[[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1],[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]];
const F=[[0,1,2,3],[7,6,5,4],[4,5,1,0],[6,7,3,2],[5,6,2,1],[4,0,3,7]];
F.forEach(f=>poly3(f.map(i=>P[i]),col));}
function drawFields(){if(!document.getElementById('ground').checked)return;
function trackPts(cx,cz,ax,bs){const P=[],n=7;
for(let k=0;k<=n;k++)P.push([cx+ax,cz-bs+2*bs*k/n]);
for(let k=0;k<n;k++){const a0=-Math.PI/2+Math.PI*k/n,a1=-Math.PI/2+Math.PI*(k+1)/n;P.push([cx+ax*Math.cos((a0+a1)/2),cz+bs+ax*Math.sin((a0+a1)/2)]);}
for(let k=0;k<=n;k++)P.push([cx-ax,cz+bs-2*bs*k/n]);
for(let k=0;k<n;k++){const a0=Math.PI/2+Math.PI*k/n,a1=Math.PI/2+Math.PI*(k+1)/n;P.push([cx+ax*Math.cos((a0+a1)/2),cz-bs+ax*Math.sin((a0+a1)/2)]);}
return P;}
const F=[[274,-478.5,65,183,47,147],[69,427,83.5,204,65.5,168]];
F.forEach(f=>{const cx=f[0],cz=f[1],ax=f[2],bz=f[3],axi=f[4],bzi=f[5];
const o=trackPts(cx,cz,ax,(bz-2*ax)/2),ii=trackPts(cx,cz,axi,(bzi-2*axi)/2);
poly3(o.map(q=>proj(q[0],.4,q[1])),'#c96a5c');poly3(ii.map(q=>proj(q[0],.45,q[1])),'#5f9e46');});
const ST=[[207,-478,70,4,3,1.2]];
ST.forEach(f=>{const xe=f[0],cz=f[1],L=f[2],T=f[3],W=f[4],BH=f[5];
for(let i=0;i<T;i++){const x1=xe-W*i,x0=x1-W,h=BH*(i+1);
const a=[x0,h,cz-L/2],b=[x1,h,cz-L/2],c=[x1,h,cz+L/2],d=[x0,h,cz+L/2];
poly3([proj(a[0],a[1],a[2]),proj(b[0],b[1],b[2]),proj(c[0],c[1],c[2]),proj(d[0],d[1],d[2])],i%2?'#cfc9bb':'#d9dde0');
poly3([proj(x1,0,cz-L/2),proj(x1,0,cz+L/2),proj(x1,h,cz+L/2),proj(x1,h,cz-L/2)],i%2?'#b5bcc4':'#c4cad2');}});}
function draw3D(){sc=baseScale()*zoom;
if(walkMode){drawWalk();return;}
fillBg();drawBase();if(document.getElementById('ground').checked)drawGround();drawFields();
const order=DATA.map(r=>({s:r.p.reduce((q,p)=>q+p[0]+p[1],0)/r.p.length,f:()=>drawRec(r)}))
.concat(TREES.map(t=>({s:t.x+t.z,f:()=>drawTree(t)}))).sort((a,b)=>a.s-b.s);
order.forEach(o=>o.f());
function drawRec(r){const p=r.p;if(p.length<3)return;
if(r.c!=='building'){return;}
const a=area(p),cx=p.reduce((s,q)=>s+q[0],0)/p.length,cz=p.reduce((s,q)=>s+q[1],0)/p.length;let h=height(a);
const rb=(cx>100&&a>=800)?'#566b7d':((a<700&&(((cx*13+cz*7)%3)+3)%3===0)?'#c27b35':BASE.roof);
const fk=Math.round(cx)+','+Math.round(cz);if(FLORTAB[fk])h=FLORTAB[fk]*3.2+1.2;
if(!walkMode)poly3(p.map(q=>[q[0]+14,.02,q[1]+10]),'rgba(52,64,58,.12)');
let dA=null,dB=null,dBest=0;
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length];
const L=Math.hypot(v[0]-u[0],v[1]-u[1]);if(L>dBest){dBest=L;dA=u;dB=v;}
const lit=.72+.28*Math.abs(Math.cos(Math.atan2(v[1]-u[1],v[0]-u[0])+ang));
const wc=BCOL[fk]||(h>=25?'#9fb2c4':BASE.building);
poly3([[u[0],0,u[1]],[v[0],0,v[1]],[v[0],h,v[1]],[u[0],h,u[1]]],sh(wc,lit));}
if(walkMode){
const ux=(dB[0]-dA[0])/dBest,uz=(dB[1]-dA[1])/dBest;
let ox=uz,oz=-ux;const mx=(dA[0]+dB[0])/2,mz=(dA[1]+dB[1])/2;
const inx=mx-ox*2,inz=mz-oz*2;if(blockedH(inx,inz,1.6)){ox=-ox;oz=-oz;}
const dw=2.4,dh=2.8;
poly3([[mx-ux*dw/2+ox*.15,0,mz-uz*dw/2+oz*.15],[mx+ux*dw/2+ox*.15,0,mz+uz*dw/2+oz*.15],[mx+ux*dw/2+ox*.15,dh,mz+uz*dw/2+oz*.15],[mx-ux*dw/2+ox*.15,dh,mz-uz*dw/2+oz*.15]],'#3c4a63',null,.6);
poly3([[mx-ux*(dw/2+1.2)+ox*1.7,0,mz-uz*(dw/2+1.2)+oz*1.7],[mx+ux*(dw/2+1.2)+ox*1.7,0,mz+uz*(dw/2+1.2)+oz*1.7],[mx+ux*(dw/2+1.2)+ox*1.7,.36,mz+uz*(dw/2+1.2)+oz*1.7],[mx-ux*(dw/2+1.2)+ox*1.7,.36,mz-uz*(dw/2+1.2)+oz*1.7]],'#cfc9bb',null,.5);
poly3([[mx-ux*(dw/2+.5)+ox*.85,0,mz-uz*(dw/2+.5)+oz*.85],[mx+ux*(dw/2+.5)+ox*.85,0,mz+uz*(dw/2+.5)+oz*.85],[mx+ux*(dw/2+.5)+ox*.85,.18,mz+uz*(dw/2+.5)+oz*.85],[mx-ux*(dw/2+.5)+ox*.85,.18,mz-uz*(dw/2+.5)+oz*.85]],'#d9dde0',null,.5);
const nfl=Math.floor(h/3.2);
for(let f=1;f<nfl;f++){const fy=f*3.2;
poly3([[mx-ux*dBest/2,fy,mz-uz*dBest/2],[mx+ux*dBest/2,fy,mz+uz*dBest/2],[mx+ux*dBest/2,fy+.22,mz+uz*dBest/2],[mx-ux*dBest/2,fy+.22,mz-uz*dBest/2]],'rgba(110,120,135,.9)',null,.4);
// 窗户：每层沿最长边分布深色窗带
for(let f=0;f<nfl;f++){const wy=f*3.2+1.4;for(let w2=0;w2<4;w2++){const t0=.12+w2*.2,t1=t0+.12;
const wx0=dA[0]+(dB[0]-dA[0])*t0-ox*.18,wz0=dA[1]+(dB[1]-dA[1])*t0-oz*.18;
const wx1=dA[0]+(dB[0]-dA[0])*t1-ox*.18,wz1=dA[1]+(dB[1]-dA[1])*t1-oz*.18;
poly3([[wx0,wy,wz0],[wx1,wy,wz1],[wx1,wy+1.1,wz1],[wx0,wy+1.1,wz0]],'rgba(86,107,125,.75)',null,.3);}}}}
poly3(p.map(q=>[q[0],h,q[1]]),sh(rb,1.06));
if(!walkMode){ctx.strokeStyle='rgba(30,38,46,.32)';ctx.lineWidth=.6;{const sp=p.map(q=>proj(q[0],h,q[1]));ctx.beginPath();sp.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.stroke();}}
const k=.12,rise=Math.min(4.0,Math.max(3.0,h*.18)),ip=p.map(q=>[cx+(q[0]-cx)*(1-k),cz+(q[1]-cz)*(1-k)]);
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length],c=ip[i],d=ip[(i+1)%ip.length];
poly3([[u[0],h,u[1]],[v[0],h,v[1]],[d[0],h+rise,d[1]],[c[0],h+rise,c[1]]],sh(rb,.82+.2*((i%3)/3)));}
poly3(ip.map(q=>[q[0],h+rise,q[1]]),sh(rb,.68));
// 屋脊线（R2效果图：坡顶有中脊；沿内缩轮廓最长方向）
if(ip.length>2&&!walkMode){let bi=0,bd=0;
  for(let i=0;i<ip.length;i++){const u=ip[i],v=ip[(i+1)%ip.length],d=Math.hypot(v[0]-u[0],v[1]-u[1]);if(d>bd){bd=d;bi=i;}}
  const u=ip[bi],v=ip[(bi+1)%ip.length];
  ctx.strokeStyle='rgba(18,24,30,.5)';ctx.lineWidth=1.4;ctx.beginPath();
  const qa=proj((u[0]+v[0])/2,h+rise,(u[1]+v[1])/2),qb=proj(cx,h+rise,cz);
  ctx.moveTo(qa[0],qa[1]);ctx.lineTo(qb[0],qb[1]);ctx.stroke();}
if(!walkMode){ctx.strokeStyle='rgba(20,26,30,.45)';ctx.lineWidth=.6;{const sp=ip.map(q=>proj(q[0],h+rise,q[1]));ctx.beginPath();sp.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.stroke();}}
drawMaotou(p,h);
}
    function solveAffine(src,dst){const[x0,y0]=src[0],[x1,y1]=src[1],[x2,y2]=src[2],[X0,Y0]=dst[0],[X1,Y1]=dst[1],[X2,Y2]=dst[2];
    const den=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0);if(Math.abs(den)<1e-9)return[1,0,0,1,0,0];
    const a=((X1-X0)*(y2-y0)-(X2-X0)*(y1-y0))/den,c=((X2-X0)*(x1-x0)-(X1-X0)*(x2-x0))/den;
    const e=X0-a*x0-c*y0,b=((Y1-Y0)*(y2-y0)-(Y2-Y0)*(y1-y0))/den,d2=((Y2-Y0)*(x1-x0)-(Y1-Y0)*(x2-x0))/den;
    return[a,b,c,d2,e,Y0-b*x0-d2*y0];}
    var GTILE=null;
    function groundPattern(){if(GTILE)return GTILE;
    const c=document.createElement('canvas');c.width=64;c.height=64;const g=c.getContext('2d');
    g.fillStyle='#93ab83';g.fillRect(0,0,64,64);
    g.strokeStyle='rgba(255,255,255,.16)';g.lineWidth=1;
    for(let k=0;k<=64;k+=16){g.beginPath();g.moveTo(k,0);g.lineTo(k,64);g.stroke();
    g.beginPath();g.moveTo(0,k);g.lineTo(64,k);g.stroke();}
    GTILE=ctx.createPattern(c,'repeat');return GTILE;}
    function drawGroundWalk(){
    // 真·透视纹理：把每个世界地面格再细分为屏幕空间小三角，逐三角用局部仿射贴图
    // 近处格细分更密 => 仿射误差可忽略 => 无条纹、纹理钉在世界坐标不随移动变
    if(!GIMG.complete||!GIMG.naturalWidth){ctx.fillStyle='#93ab83';ctx.fillRect(0,cv.height*.3,cv.width,cv.height);return;}
    const U=wx=>(wx+497.24)*1.81,V=wz=>(wz+661.05)*1.81;
    const tris=[];
    const R=260,T=20,SUB=4; // 固定4x4剖分：世界坐标对齐，剖分不随相机变，纹理钉死不动
    const cx0=Math.max(-520,Math.floor((camX-R)/T)*T),cx1=Math.min(500,camX+R);
    const cz0=Math.max(-660,Math.floor((camZ-R)/T)*T),cz1=Math.min(640,camZ+R);
    for(let gx=cx0;gx<cx1;gx+=T)for(let gz=cz0;gz<cz1;gz+=T){
    const d=camD(gx+T/2,gz+T/2);if(d<0.6||d>R*1.6)continue;
    const sub=SUB;
    const step=T/sub;
    for(let ax=0;ax<sub;ax++)for(let az=0;az<sub;az++){
    const x0=gx+ax*step,z0=gz+az*step,x1=x0+step,z1=z0+step;
    const corners=[[x0,z0],[x1,z0],[x1,z1],[x0,z1]];
    const cs=corners.map(p=>camSpace(p[0],0,p[1]));
    if(cs.some(q=>q[1]<0.35))continue;
    const P=cs.map(q=>wproj(q));
    if(P.every(p=>p[0]<-100||p[0]>cv.width+100||p[1]<-100||p[1]>cv.height+100))continue;
    const uv=corners.map(p=>[U(p[0]),V(p[1])]);
    const dd=cs.reduce((s2,q)=>s2+q[1],0)/4;
    tris.push({P,uv,d:dd});}}
    tris.sort((a,b)=>b.d-a.d);
    tris.forEach(t=>{const m=solveAffine(t.uv.slice(0,3),t.P.slice(0,3));
    ctx.save();ctx.beginPath();t.P.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.clip();
    ctx.transform(m[0],m[1],m[2],m[3],m[4],m[5]);ctx.drawImage(GIMG,0,0);ctx.restore();});
    // 脚下近处（相机前 0~4m）无贴图区域用纯色兜底
    const gg=ctx.createLinearGradient(0,cv.height*.84,0,cv.height);gg.addColorStop(0,'rgba(147,171,131,0)');gg.addColorStop(1,'#93ab83');ctx.fillStyle=gg;ctx.fillRect(0,cv.height*.84,cv.width,cv.height*.16);
    }
function drawWalk(){
    walkFaces=[];
    ctx.fillStyle='#a8c8e0';ctx.fillRect(0,0,cv.width,cv.height);
    if(document.getElementById('ground').checked&&GIMG.complete&&GIMG.naturalWidth)drawGroundWalk();
    else wpoly([[-520,0,-665],[520,0,-665],[520,0,665],[-520,0,665]],'#7fa85a');
    if(document.getElementById('ground').checked){DATA.filter(r=>r.c==='water').forEach(r=>wpoly(r.p.map(q=>[q[0],0.14,q[1]]),'#6fa8cf'));}
    DETAILS.forEach(d=>drawDetail(d));
    DATA.filter(r=>r.c==='building').forEach(r=>drawRec(r));
    walkFaces.sort((a,b)=>b.d-a.d);
    walkFaces.forEach(f=>{ctx.beginPath();f.P.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=f.col;ctx.fill();if(f.st){ctx.strokeStyle=f.st;ctx.lineWidth=.5;ctx.stroke();}});
    if(document.getElementById('tree').checked){TREES.map(t=>({t:t,d:camD(t.x,t.z)})).filter(x=>x.d>.8&&x.d<420).sort((a,b)=>b.d-a.d).forEach(x=>drawTree(x.t));}
    ctx.fillStyle='rgba(255,255,255,.88)';ctx.fillRect(10,cv.height-66,340,54);
    ctx.fillStyle='#1f2328';ctx.font='12px "Microsoft YaHei",sans-serif';ctx.textAlign='left';
    ctx.fillText('W/S 前后　A/D 平移　空格/C 升降　拖动转向(上下=抬头低头)',18,cv.height-48);
    ctx.fillText('Esc 退出漫游　碰撞阻挡已开启',18,cv.height-28);
    ctx.fillStyle='rgba(255,255,255,.55)';ctx.beginPath();ctx.arc(cv.width/2,cv.height/2,2.5,0,7);ctx.fill();
    if(GIMG.complete&&GIMG.naturalWidth){
    const mw=mapZoom?Math.min(680,cv.width-40):260,mh=mw*2395/1800;
    const mx=mapZoom?Math.round((cv.width-mw)/2):cv.width-mw-14,my=mapZoom?Math.round((cv.height-mh)/2):14;
    ctx.fillStyle='rgba(255,255,255,.95)';ctx.fillRect(mx-6,my-30,mw+12,mh+12);
    ctx.drawImage(GIMG,mx,my,mw,mh);
    const px=mx+(camX+500)/1000*mw,py=my+(camZ+661)/1322*mh;
    // 视角箭头：尖端指向视线方向（随camYaw），尾翼展开
    const AL=mapZoom?80:52,ax=Math.cos(camYaw),ay=Math.sin(camYaw);
    const qx=px+ax*10*(mw/1000),qy=py+ay*10*(mh/1322); // 箭尾从点外10m起，不与点重叠
    const ex=qx+ax*AL*(mw/1000),ey=qy+ay*AL*(mh/1322);
    const bx=qx-ax*AL*.15*(mw/1000),by=qy-ay*AL*.15*(mh/1322);
    const wx=-ay*(mapZoom?18:14)*(mw/1000),wy=ax*(mapZoom?18:14)*(mh/1322);
    ctx.lineWidth=mapZoom?3:2.5;ctx.strokeStyle='#7a1010';ctx.fillStyle='#ff2a1a';ctx.beginPath();
    ctx.moveTo(ex,ey);ctx.lineTo(bx+wx,by+wy);ctx.lineTo(bx-wx,by-wy);ctx.closePath();ctx.fill();ctx.stroke();
    ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(px,py,mapZoom?9:6,0,7);ctx.fill();
    ctx.strokeStyle='#e03c31';ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(px,py,mapZoom?9:6,0,7);ctx.stroke();
    ctx.strokeStyle='rgba(31,111,235,.65)';ctx.lineWidth=1;ctx.strokeRect(mx-6,my-30,mw+12,mh+12);
    ctx.fillStyle='#1f2328';ctx.font='600 13px "Microsoft YaHei",sans-serif';ctx.textAlign='left';
    ctx.fillText(mapZoom?'点击地图选择传送地点（点图外/Esc 取消）':'点击小地图放大并选择传送点',mx,my-10);}}

DETAILS.forEach(d=>drawDetail(d));
drawLabels();
}
function draw2D(){ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#eef3f7';ctx.fillRect(0,0,cv.width,cv.height);
const scm=(cv.width*0.9)/994.5,ox=cv.width/2,oy=cv.height/2;
const W=994.5*scm,Hh=1322.1*scm;
if(GIMG.complete&&GIMG.naturalWidth){ctx.drawImage(GIMG,ox-497.24*scm,oy-661.05*scm,W,Hh);}
const S=(x,z)=>[ox+x*scm,oy+z*scm];
DATA.forEach(r=>{const p=r.p;if(p.length<3)return;ctx.beginPath();
p.forEach((q,i)=>{const s=S(q[0],q[1]);i?ctx.lineTo(s[0],s[1]):ctx.moveTo(s[0],s[1]);});ctx.closePath();
const c={building:'rgba(235,237,235,.94)',water:'rgba(70,140,200,.38)',green:'rgba(96,164,96,.26)',road:'rgba(150,150,150,.14)'}[r.c];
ctx.fillStyle=c;ctx.fill();ctx.strokeStyle=r.c==='building'?'rgba(38,50,60,.92)':'rgba(30,90,140,.7)';ctx.lineWidth=r.c==='building'?1.6:1;ctx.stroke();});
if(document.getElementById('tree').checked){ctx.fillStyle='rgba(72,116,62,.8)';
TREES.forEach(t=>{const s=S(t.x,t.z);ctx.beginPath();ctx.arc(s[0],s[1],1.6,0,7);ctx.fill();});}
ctx.font='600 15px "Microsoft YaHei",sans-serif';ctx.textAlign='center';
LABELS.forEach(L=>{const s=S(L.x,L.z);const tw=ctx.measureText(L.n).width;
ctx.fillStyle='rgba(255,255,255,.9)';ctx.fillRect(s[0]-tw/2-6,s[1]-18,tw+12,20);
ctx.strokeStyle='rgba(31,111,235,.7)';ctx.lineWidth=1;ctx.strokeRect(s[0]-tw/2-6,s[1]-18,tw+12,20);
ctx.fillStyle='#1f2328';ctx.fillText(L.n,s[0],s[1]-3);});
const bl=100*scm;ctx.strokeStyle='#333';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(20,cv.height-26);ctx.lineTo(20+bl,cv.height-26);ctx.stroke();
ctx.fillStyle='#333';ctx.font='12px sans-serif';ctx.textAlign='left';ctx.fillText('100 m',20,cv.height-32);
ctx.textAlign='left';ctx.fillStyle='#1f2328';ctx.font='700 18px "Microsoft YaHei",sans-serif';
ctx.fillText('南京晓庄学院（方山校区）· 俯视总平面',18,26);
ctx.font='12px "Microsoft YaHei",sans-serif';ctx.fillStyle='#57606a';ctx.fillText('真实规划图 + 楼栋/水体/绿地轮廓（示意）',18,44);
const nx=cv.width-34,ny=34;ctx.strokeStyle='#333';ctx.fillStyle='#333';ctx.lineWidth=2;
ctx.beginPath();ctx.moveTo(nx,ny+14);ctx.lineTo(nx,ny-10);ctx.stroke();
ctx.beginPath();ctx.moveTo(nx-5,ny-3);ctx.lineTo(nx,ny-12);ctx.lineTo(nx+5,ny-3);ctx.closePath();ctx.fill();
ctx.font='11px sans-serif';ctx.textAlign='center';ctx.fillText('N',nx,ny+26);}
function draw(){if(view===2)draw2D();else draw3D();}
function resetView(){ang=-.62;zoom=1;panX=0;panY=140;draw();}
function setView(v){view=v;document.getElementById('ctl3d').style.display=(v===3||v===4)?'block':'none';
if(v===4){ang=-.72;zoom=1.85;panX=-60;panY=180;} // R2效果图式近景斜俯视
draw();}
cv.addEventListener('mousedown',e=>{const[mousemx,mousemy]=mouseXY(e),mx=mousemx,my=mousemy;
if(walkMode){
const mw=mapZoom?Math.min(680,cv.width-40):260,mh=mw*2395/1800;
const mx0=mapZoom?Math.round((cv.width-mw)/2):cv.width-mw-14,my0=mapZoom?Math.round((cv.height-mh)/2):14;
const inMap=mx>=mx0&&mx<=mx0+mw&&my>=my0&&my<=my0+mh;
if(inMap&&!mapZoom){mapZoom=true;draw();return;}
if(inMap&&mapZoom){
const tx=(mx-mx0)/mw*1000-500,tz=(my-my0)/mh*1322-661;
let bx=tx,bz=tz;if(blockedH(bx,bz,camH)){let best=null;
for(let rr2=6;rr2<=80&&!best;rr2+=6){for(let a2=0;a2<12&&!best;a2++){const ax2=tx+rr2*Math.cos(a2*Math.PI/6),az2=tz+rr2*Math.sin(a2*Math.PI/6);
if(!blockedH(ax2,az2,camH))best=[ax2,az2];}}
if(best){bx=best[0];bz=best[1];}else{bx=camX;bz=camZ;}}
camX=bx;camZ=bz;mapZoom=false;draw();return;}
if(mapZoom){mapZoom=false;draw();return;}
drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';return;}
drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=false;cv.style.cursor='grab';});
window.addEventListener('mousemove',e=>{const mm=mouseXY(e);if(walkMode){if(drag){camYaw+=(mm[0]-lx)*.006;
camPitch=Math.max(-1.25,Math.min(1.25,camPitch-(mm[1]-ly)*.004));draw();}lx=mm[0];ly=mm[1];return;}
if(!drag)return;ang+=(mm[0]-lx)*.008;panY+=(mm[1]-ly)*.5;lx=mm[0];ly=mm[1];draw();});
cv.addEventListener('wheel',e=>{if(walkMode)return;e.preventDefault();zoom*=e.deltaY>0?.9:1.1;draw();},{passive:false});
window.addEventListener('keydown',e=>{if(!walkMode)return;const k=e.key.toLowerCase();walkKeys[k]=true;
if(['arrowup','arrowdown','arrowleft','arrowright',' '].includes(k))e.preventDefault();
if(k==='escape'){if(mapZoom){mapZoom=false;draw();}else toggleWalk();}});
window.addEventListener('keyup',e=>{walkKeys[e.key.toLowerCase()]=false;});
draw();</script></body></html>'''
    all_pts = [q for p in polys for q in p["points"]]
    span = max(max(q[0] for q in all_pts) - min(q[0] for q in all_pts),
               max(q[1] for q in all_pts) - min(q[1] for q in all_pts), 1.0)
    init_sc = round(336.0 / span, 5)
    html = (html.replace("__PLAN__", imgs["plan"]).replace("__OVERLAY__", imgs["overlay"])
                .replace("__REF__", imgs["ref"]).replace("__COUNT__", str(len(polys)))
                .replace("__INITSC__", str(init_sc))
                .replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
                .replace("__DETAILS__", json.dumps(det, ensure_ascii=False))
                .replace("__DATA__", json.dumps(data, ensure_ascii=False))
                .replace("__BCOL__", json.dumps(cmap, ensure_ascii=False))
                .replace("__TREES__", json.dumps(trees or [], ensure_ascii=False)))
    open(PREVIEW, "w", encoding="utf-8", newline="\n").write(html)
    return len(html.encode("utf-8"))


def main():
    d = json.load(open(OUTLINES, encoding="utf-8"))
    polys = d["polygons"]
    textures = make_textures()  # 仍生成 docs/materials 纹理源文件作参考
    ground = load_ground()
    details = compute_details(polys)
    labels = compute_labels(polys)
    parts = build_parts(polys, details)
    # 树：位置取自规划图树符号（extract_trees.py）
    trees = []
    if os.path.exists(TREES_PATH):
        trees = json.load(open(TREES_PATH, encoding="utf-8"))["trees"]
        for t in trees:
            parts.extend(tree_geometry(t))
    size, tris, prims = build_glb(parts, ground, textures)
    html_size = make_preview(polys, details, labels, ground, trees)
    rep = {"stage": "徽派风格（青砖小瓦马头墙）+ 真实规划图地面 + 树木 + 中文标注",
           "source": "官方总平面图真实轮廓 + 真实规划图地面贴图 + 树符号位置",
           "polygons": len(polys), "scene_details": len(details), "trees": len(trees),
           "primitives": prims, "triangles": tris,
           "ground_map": "refs/official_01_plan_d.jpg (1800x2395)",
           "glb_bytes": size, "preview_bytes": html_size,
           "note": "地面铺真实规划图（与轮廓同一坐标系）；建筑按徽派程序化上色（青砖墙纹理/小瓦顶纹理/马头墙黛瓦山墙）；坐标由图上 100m 比例尺定标"}
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("材质模型:", size, "bytes | 三角面", tris, "| 地面贴图", len(ground), "bytes")
    print("离线预览:", html_size, "bytes")


if __name__ == "__main__":
    main()
