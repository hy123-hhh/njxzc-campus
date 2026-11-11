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

# 扁平建筑配色（干净、可读，不依赖噪点纹理）
FLAT = {
    "building": (236, 238, 239), "roof": (58, 72, 99),
    "water": (111, 168, 207), "green": (127, 168, 90),
    "road": (201, 200, 192),
}
# 规划图在 small(1800x2395) 像素空间下的尺寸与比例尺（与 refine_geometry 一致）
PLAN_W, PLAN_H, PX_PER_M = 1800, 2395, 1.81


def make_textures():
    os.makedirs(MATDIR, exist_ok=True)
    rng = np.random.default_rng(20261111)
    out = {}
    S = 256

    roof = np.zeros((S, S, 3), np.uint8)
    roof[:] = (99, 74, 60)
    roof = (roof.astype(np.int16) + rng.normal(0, 5, (S, S, 1))).clip(0, 255).astype(np.uint8)
    for y in range(0, S, 18):
        cv2.line(roof, (0, y), (S, y), (48, 58, 78), 2)
        off = 0 if (y // 18) % 2 == 0 else 16
        for x in range(off, S, 32):
            cv2.line(roof, (x, y), (x, y + 18), (48, 58, 78), 1)
    out["roof"] = roof

    wall = np.full((S, S, 3), 234, np.uint8)
    wall = (wall.astype(np.int16) + rng.normal(0, 3, (S, S, 1))).clip(0, 255).astype(np.uint8)
    for y in range(0, S, 40):
        cv2.line(wall, (0, y), (S, y), (198, 202, 205), 2)
        for x in range(8, S, 24):
            cv2.rectangle(wall, (x, y + 6), (x + 14, y + 28), (150, 178, 198), -1)
            cv2.rectangle(wall, (x, y + 6), (x + 14, y + 28), (120, 140, 158), 1)
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
    """读取官方规划图，缩放到与轮廓提取一致的 1800x2395 像素空间，作为 3D 地面。"""
    p = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return b""
    small = cv2.resize(img, (PLAN_W, PLAN_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes() if ok else b""


def compute_labels(polys):
    """主要功能区中文标注（按真实布局锚定）。"""
    labels = []
    bs = [p for p in polys if p["class"] == "building"]
    if bs:
        b = max(bs, key=lambda p: p.get("area_px", 0))
        labels.append({"n": "图书馆", "x": b["centroid"][0], "z": b["centroid"][1]})
    waters = [p for p in polys if p["class"] == "water"]
    if waters:
        w = max(waters, key=lambda p: p.get("area_px", 0))
        labels.append({"n": "人工湖", "x": w["centroid"][0], "z": w["centroid"][1]})
    labels += [
        {"n": "陶行知像", "x": -60, "z": -30},
        {"n": "北运动场", "x": 300, "z": -380},
        {"n": "南运动场", "x": 100, "z": 430},
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
    bs = sorted([p for p in polys if p["class"] == "building"],
                key=lambda p: -p.get("area_px", 0))
    if bs:
        c = bs[0]["centroid"]
        add("tower", c[0] - 52, c[1], 18, 18, 40, "#d9dde0", "图书馆西塔")
        add("tower", c[0] + 52, c[1], 18, 18, 40, "#b8cdd9", "图书馆东塔")
    add("treeline", -400, 0, 10, 900, 11, "#d9a441", "西门银杏树列")
    add("treeline", 230, 60, 10, 700, 10, "#e3b7c4", "樱花大道树列")
    add("flowerbed", -200, -30, 12, 260, 0.8, "#c04b40", "中轴红色花带西")
    add("flowerbed", 80, -30, 12, 260, 0.8, "#c04b40", "中轴红色花带东")
    add("track", 300, -380, 190, 130, 0.4, "#b5544a", "北操场跑道环")
    add("track", 100, 430, 190, 140, 0.4, "#b5544a", "南操场跑道环")
    add("rockery", -400, -200, 50, 30, 5, "#b9b5a9", "西门假山")
    add("monument", -430, 180, 24, 9, 6, "#a9a49a", "校名石")
    add("bridge", 100, -20, 70, 9, 2, "#cfc9bb", "湖上石桥")
    return d


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
    for p in polys:
        cls, pts = p["class"], p["points"]
        if len(pts) < 3:
            continue
        tris = tris_of(pts)
        if not tris:
            continue
        if cls != "building":
            m = TMesh()
            y = {"water": 0.14, "green": 0.10, "road": 0.20}[cls]
            for t in tris:
                m.tri([(a, y, b) for a, b in t], (0, 1, 0),
                      [uv_top(a, b) for a, b in t])
            parts.append((m, cls))
            continue
        area = abs(Polygon(pts).area)
        h = building_height(area)
        body = TMesh()
        for t in tris:
            body.tri([(a, h, b) for a, b in t], (0, 1, 0),
                     [uv_top(a, b) for a, b in t])
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            length = max(math.dist(a, b), 1.0)
            body.quad([(a[0], 0, a[1]), (b[0], 0, b[1]),
                       (b[0], h, b[1]), (a[0], h, a[1])],
                      (a[1] - b[1], 0, b[0] - a[0]),
                      [(0, 0), (length / 14, 0), (length / 14, h / 14), (0, h / 14)])
        parts.append((body, "building"))
        roof = TMesh()
        try:
            inner = Polygon(pts).buffer(-2.6, join_style=2)
            if not inner.is_empty and inner.geom_type == "Polygon":
                ip = list(inner.exterior.coords)[:-1]
                if len(ip) >= 3:
                    rise = min(6.0, max(2.5, h * 0.18))
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
            parts.append((roof, "roof"))
    for item in (details or []):
        parts.extend(detail_geometry(item))
    return parts


def build_glb(parts, ground):
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


def make_preview(polys, details, labels, ground):
    data = [{"c": p["class"], "p": p["points"]} for p in polys]
    det = [{"t": d["t"], "x": d["x"], "z": d["z"], "w": d["w"],
            "d": d["d"], "h": d["h"], "c": d["c"], "n": d["n"]} for d in details]
    ground_b64 = base64.b64encode(ground).decode() if ground else ""
    imgs = {"plan": ground_b64,
            "overlay": b64(os.path.join(ROOT, "docs", "outline_overlay.jpg")),
            "aerial": b64(os.path.join(ROOT, "refs", "places", "aerial_01.jpg"))}
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
<p class="sub">几何形状逐像素取自官方总平面图，坐标由图上 100m 比例尺定标；地面直接铺真实规划图（与轮廓同一坐标系，俯视即为校园地图），并标注主要功能区。</p>
<div class="status">本页<b>自包含、不联网</b>。左下角「规划总平面图」与三维场景地面是同一张图。<br>材质为扁平配色示意；坐标为线性映射示意值，非测绘成果。</div>
<div class="split">
<div class="pane"><h3>官方规划总平面图（地面底图）</h3><img id="planimg" src="data:image/jpeg;base64,__PLAN__" alt="总平面图"></div>
<div class="pane"><h3>云端描边验收</h3><img src="data:image/jpeg;base64,__OVERLAY__" alt="描边验收"></div>
<div class="pane"><h3>四大主楼实拍（配色依据）</h3><img src="data:image/jpeg;base64,__AERIAL__" alt="实拍"></div>
</div>
<h2>校园地图与三维预览</h2>
<div class="ctl"><button onclick="setView(2)">俯视地图</button> <button onclick="setView(3)">三维预览</button>
<span id="hint">默认俯视图：真实规划图 + 楼栋填色 + 中文标注 + 100m 比例尺</span></div>
<canvas id="cv" width="1000" height="600"></canvas>
<div class="ctl" id="ctl3d" style="display:none">左右拖动旋转，上下平移，滚轮缩放　<button onclick="resetView()">重置视角</button>
<label><input type="checkbox" id="ground" checked onchange="draw()"> 规划图地面</label>
<label><input type="checkbox" id="label" checked onchange="draw()"> 标注</label>
<label><input type="checkbox" id="tree" checked onchange="draw()"> 树木</label></div>
<div class="note">共 __COUNT__ 个真实轮廓。来源：学校发展规划处公开总平面规划图；实拍来自公开媒体转载，版权归原发布方。教学展示用途 · 示意性布局 · 非测绘成果。</div>
<script>
const DATA=__DATA__;
const DETAILS=__DETAILS__;
const LABELS=__LABELS__;
const GROUND=document.getElementById('planimg').src.split(',')[1];
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const GIMG=new Image();GIMG.onload=draw;GIMG.src='data:image/jpeg;base64,'+GROUND;
const BASE={building:'#e7e9ea',roof:'#3c4a63',water:'#6fa8cf',green:'#7fa85a',road:'#c9c8c0'};
let ang=-.62,sc=__INITSC__,panX=0,panY=140,drag=false,lx=0,ly=0,view=2;
function sh(h,f){const s=h.replace('#','');return 'rgb('+[0,2,4].map(i=>Math.min(255,parseInt(s.slice(i,i+2),16)*f|0)).join(',')+')';}
function proj(x,y,z){const c=Math.cos(ang),s=Math.sin(ang),rx=x*c-z*s,rz=x*s+z*c;
return [cv.width/2+(rx-rz)*.866*sc+panX,cv.height/2+(rx+rz)*.5*sc-y*sc*.9+panY];}
function poly3(pts,fill,stroke){ctx.beginPath();pts.forEach((p,i)=>{const q=proj(p[0],p[1],p[2]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke||'rgba(35,45,60,.25)';ctx.lineWidth=.5;ctx.stroke();}
function area(p){let a=0;for(let i=0;i<p.length;i++){const q=p[(i+1)%p.length];a+=p[i][0]*q[1]-q[0]*p[i][1];}return Math.abs(a)/2;}
function height(a){return a>9000?21:(a>3500?18:(a>1200?15:11));}
function drawDetail(d){
if(d.t==='treeline'){const n=Math.max(6,Math.floor(d.d/45));
for(let i=0;i<n;i++){const zz=d.z-d.d/2+d.d*(i+.5)/n,q=proj(d.x,d.h*.62,zz);
ctx.beginPath();ctx.arc(q[0],q[1],4.6,0,7);ctx.fillStyle=d.c;ctx.fill();
ctx.beginPath();ctx.arc(q[0],q[1]+5,1.8,0,7);ctx.fillStyle='#8a7256';ctx.fill();}return;}
if(d.t==='track'){const x0=d.x-d.w/2,x1=d.x+d.w/2,z0=d.z-d.d/2,z1=d.z+d.d/2;
ctx.beginPath();[[x0,.3,z0],[x1,.3,z0],[x1,.3,z1],[x0,.3,z1],[x0,.3,z0]].forEach((p,i)=>{const q=proj(p[0],p[1],p[2]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
ctx.strokeStyle=d.c;ctx.lineWidth=3;ctx.stroke();return;}
if(d.t==='statue'||d.t==='roundbed'||d.t==='pavilion'||d.t==='tower'){
const r=Math.min(d.w,d.d)/2,q0=proj(d.x,0,d.z),q1=proj(d.x,d.h,d.z),w=r*sc;
ctx.beginPath();ctx.moveTo(q0[0]-w,q0[1]);ctx.lineTo(q0[0]+w,q0[1]);ctx.lineTo(q1[0]+w,q1[1]);ctx.lineTo(q1[0]-w,q1[1]);ctx.closePath();
ctx.fillStyle=d.c;ctx.fill();ctx.strokeStyle='rgba(40,50,60,.35)';ctx.lineWidth=.6;ctx.stroke();
if(d.t==='pavilion'||d.t==='tower'){const t=proj(d.x,d.h+d.h*.42,d.z);
ctx.beginPath();ctx.moveTo(q1[0]-w*1.5,q1[1]);ctx.lineTo(t[0],t[1]);ctx.lineTo(q1[0]+w*1.5,q1[1]);ctx.closePath();ctx.fillStyle='#3c4a63';ctx.fill();}return;}
const x0=d.x-d.w/2,x1=d.x+d.w/2,z0=d.z-d.d/2,z1=d.z+d.d/2;
poly3([[x0,0,z0],[x1,0,z0],[x1,d.h,z0],[x0,d.h,z0]],d.c);
poly3([[x0,d.h,z0],[x1,d.h,z0],[x1,d.h,z1],[x0,d.h,z1]],d.c);
}
function fillBg(){ctx.setTransform(1,0,0,1,0,0);const g=ctx.createLinearGradient(0,0,0,cv.height);g.addColorStop(0,'#dfeaf5');g.addColorStop(.55,'#eef3f7');g.addColorStop(1,'#e7eddf');ctx.fillStyle=g;ctx.fillRect(0,0,cv.width,cv.height);}
function groundMatrix(){const W=1800,H=2395,ppm=1.81;const wx=u=>(u-900)/ppm,wz=v=>(v-1197.5)/ppm;
const s00=proj(wx(0),0,wz(0)),s10=proj(wx(W),0,wz(0)),s01=proj(wx(0),0,wz(H));
return {a:(s10[0]-s00[0])/W,b:(s10[1]-s00[1])/W,c:(s01[0]-s00[0])/H,d:(s01[1]-s00[1])/H,e:s00[0],f:s00[1]};}
function drawGround(){if(!GIMG.complete||!GIMG.naturalWidth)return;const m=groundMatrix();
ctx.save();ctx.setTransform(m.a,m.b,m.c,m.d,m.e,m.f);ctx.drawImage(GIMG,0,0);ctx.restore();ctx.setTransform(1,0,0,1,0,0);}
function drawLabels(){if(!document.getElementById('label').checked)return;
ctx.font='600 13px "Microsoft YaHei",sans-serif';ctx.textAlign='center';
LABELS.forEach(L=>{const q=proj(L.x,8,L.z);const tw=ctx.measureText(L.n).width;
ctx.fillStyle='rgba(255,255,255,.88)';ctx.fillRect(q[0]-tw/2-5,q[1]-16,tw+10,18);
ctx.strokeStyle='rgba(31,111,235,.65)';ctx.lineWidth=1;ctx.strokeRect(q[0]-tw/2-5,q[1]-16,tw+10,18);
ctx.fillStyle='#1f2328';ctx.fillText(L.n,q[0],q[1]-3);});}
function draw3D(){fillBg();if(document.getElementById('ground').checked)drawGround();const trees=document.getElementById('tree').checked;
const order=DATA.slice().sort((a,b)=>{const ca=a.p.reduce((s,p)=>s+p[0]+p[1],0)/a.p.length,cb=b.p.reduce((s,p)=>s+p[0]+p[1],0)/b.p.length;return ca-cb;});
order.forEach(r=>{const p=r.p;if(p.length<3)return;
if(r.c!=='building'){const y=r.c==='water'?.14:(r.c==='green'?.1:.2);
poly3(p.map(q=>[q[0],y,q[1]]),BASE[r.c]);
if(r.c==='green'&&trees){const cx=p.reduce((s,q)=>s+q[0],0)/p.length,cz=p.reduce((s,q)=>s+q[1],0)/p.length;
for(let i=0;i<Math.min(10,Math.max(2,Math.floor(area(p)/900)));i++){const a=i*2.4;const q=proj(cx+Math.cos(a)*9,3,cz+Math.sin(a)*9);
ctx.beginPath();ctx.arc(q[0],q[1],3.2,0,7);ctx.fillStyle='rgba(72,116,62,.75)';ctx.fill();}}
return;}
const a=area(p),h=height(a),cx=p.reduce((s,q)=>s+q[0],0)/p.length,cz=p.reduce((s,q)=>s+q[1],0)/p.length;
poly3(p.map(q=>[q[0]+14,.02,q[1]+10]),'rgba(52,64,58,.12)');
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length];
const lit=.72+.28*Math.abs(Math.cos(Math.atan2(v[1]-u[1],v[0]-u[0])+ang));
poly3([[u[0],0,u[1]],[v[0],0,v[1]],[v[0],h,v[1]],[u[0],h,u[1]]],sh(BASE.building,lit));}
poly3(p.map(q=>[q[0],h,q[1]]),sh(BASE.building,1.06));
const k=.12,rise=Math.min(6,Math.max(2.5,h*.18)),ip=p.map(q=>[cx+(q[0]-cx)*(1-k),cz+(q[1]-cz)*(1-k)]);
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length],c=ip[i],d=ip[(i+1)%ip.length];
poly3([[u[0],h,u[1]],[v[0],h,v[1]],[d[0],h+rise,d[1]],[c[0],h+rise,c[1]]],sh(BASE.roof,.82+.2*((i%3)/3)));}
poly3(ip.map(q=>[q[0],h+rise,q[1]]),sh(BASE.roof,1.15));
});
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
const c={building:'rgba(70,90,120,.16)',water:'rgba(70,140,200,.38)',green:'rgba(96,164,96,.26)',road:'rgba(150,150,150,.14)'}[r.c];
ctx.fillStyle=c;ctx.fill();ctx.strokeStyle=r.c==='building'?'rgba(40,60,90,.7)':'rgba(30,90,140,.7)';ctx.lineWidth=1;ctx.stroke();});
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
function resetView(){ang=-.62;sc=__INITSC__;panX=0;panY=140;draw();}
function setView(v){view=v;document.getElementById('ctl3d').style.display=v===3?'block':'none';draw();}
cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=false;cv.style.cursor='grab';});
window.addEventListener('mousemove',e=>{if(!drag)return;ang+=(e.clientX-lx)*.008;panY+=(e.clientY-ly)*.5;lx=e.clientX;ly=e.clientY;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();sc*=e.deltaY>0?.9:1.1;draw();},{passive:false});
draw();</script></body></html>'''
    all_pts = [q for p in polys for q in p["points"]]
    span = max(max(q[0] for q in all_pts) - min(q[0] for q in all_pts),
               max(q[1] for q in all_pts) - min(q[1] for q in all_pts), 1.0)
    init_sc = round(336.0 / span, 5)
    html = (html.replace("__PLAN__", imgs["plan"]).replace("__OVERLAY__", imgs["overlay"])
                .replace("__AERIAL__", imgs["aerial"]).replace("__COUNT__", str(len(polys)))
                .replace("__INITSC__", str(init_sc))
                .replace("__LABELS__", json.dumps(labels, ensure_ascii=False))
                .replace("__DETAILS__", json.dumps(det, ensure_ascii=False))
                .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
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
    size, tris, prims = build_glb(parts, ground)
    html_size = make_preview(polys, details, labels, ground)
    rep = {"stage": "真实规划图地面 + 扁平配色 + 中文标注",
           "source": "官方总平面图真实轮廓 + 真实规划图地面贴图",
           "polygons": len(polys), "scene_details": len(details),
           "primitives": prims, "triangles": tris,
           "ground_map": "refs/official_01_plan_d.jpg (1800x2395)",
           "glb_bytes": size, "preview_bytes": html_size,
           "note": "地面铺真实规划图（与轮廓同一坐标系）；建筑为扁平配色示意；坐标由图上 100m 比例尺定标"}
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("材质模型:", size, "bytes | 三角面", tris, "| 地面贴图", len(ground), "bytes")
    print("离线预览:", html_size, "bytes")


if __name__ == "__main__":
    main()
