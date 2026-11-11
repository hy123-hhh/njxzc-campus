# -*- coding: utf-8 -*-
"""方山校区布局级模型生成器（纯 Python 标准库，零第三方依赖）。

从 data/layout.csv 读取布局元素，按实拍参考生成 glTF 2.0 二进制 GLB。
仅供 GitHub Actions 云端运行；本地不执行建模任务。

v2 增加：坡屋顶/攒尖顶/穹顶、楼层窗带、塔楼、柱廊、台阶、陶像、
湖心亭、树列、花带、跑道环等细节，配色依据 refs/places/ 实拍照片。

输出：
  models/campus_layout.glb
  models/layout_report.json
"""
import csv
import json
import math
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "layout.csv")
MODEL_DIR = os.path.join(ROOT, "models")
GLB_PATH = os.path.join(MODEL_DIR, "campus_layout.glb")
REPORT_PATH = os.path.join(MODEL_DIR, "layout_report.json")

ROOF_COLOR = "#3c4a63"
GLASS_COLOR = "#8fb3ce"
STEP_COLOR = "#d9d8d2"
TREE_TRUNK = "#8a7256"


def hex_rgb(value):
    s = (value or "#cccccc").lstrip("#")
    return [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


class Mesh:
    def __init__(self):
        self.pos = []
        self.nrm = []
        self.idx = []

    def quad(self, points, normal):
        base = len(self.pos)
        for p in points:
            self.pos.append(p)
            self.nrm.append(normal)
        self.idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    def tri(self, points, normal):
        base = len(self.pos)
        for p in points:
            self.pos.append(p)
            self.nrm.append(normal)
        self.idx.extend([base, base + 1, base + 2])

    def box(self, x, z, w, d, h, base=0.0):
        x0, x1 = x - w / 2, x + w / 2
        z0, z1 = z - d / 2, z + d / 2
        y0, y1 = base, base + h
        self.quad([(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)], (0, 1, 0))
        self.quad([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0))
        self.quad([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)], (0, 0, -1))
        self.quad([(x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)], (0, 0, 1))
        self.quad([(x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0)], (-1, 0, 0))
        self.quad([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0))

    def plane(self, x, z, w, d, y):
        x0, x1 = x - w / 2, x + w / 2
        z0, z1 = z - d / 2, z + d / 2
        self.quad([(x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0)], (0, 1, 0))

    def cylinder(self, x, z, radius, h, segments=20, base=0.0):
        y0, y1 = base, base + h
        for i in range(segments):
            a0 = 2 * math.pi * i / segments
            a1 = 2 * math.pi * (i + 1) / segments
            p0 = (x + radius * math.cos(a0), y1, z + radius * math.sin(a0))
            p1 = (x + radius * math.cos(a1), y1, z + radius * math.sin(a1))
            self.tri([(x, y1, z), p0, p1], (0, 1, 0))
            q0, q1 = (p0[0], y0, p0[2]), (p1[0], y0, p1[2])
            self.quad([q0, p0, p1, q1],
                      (math.cos((a0 + a1) / 2), 0, math.sin((a0 + a1) / 2)))

    def cone(self, x, z, radius, h, segments=12, base=0.0):
        for i in range(segments):
            a0 = 2 * math.pi * i / segments
            a1 = 2 * math.pi * (i + 1) / segments
            p0 = (x + radius * math.cos(a0), base, z + radius * math.sin(a0))
            p1 = (x + radius * math.cos(a1), base, z + radius * math.sin(a1))
            self.tri([(x, base + h, z), p0, p1],
                     (math.cos((a0 + a1) / 2), 0.6, math.sin((a0 + a1) / 2)))

    def hip_roof(self, x, z, w, d, y, rise):
        """四坡顶：底四角收拢到顶部小矩形，实拍四大主楼/宿舍的深蓝坡屋顶。"""
        x0, x1, z0, z1 = x - w / 2, x + w / 2, z - d / 2, z + d / 2
        sx, sz = w * 0.30, d * 0.30
        a0, a1, b0, b1 = x - sx / 2, x + sx / 2, z - sz / 2, z + sz / 2
        t = y + rise
        self.quad([(x0, y, z0), (x1, y, z0), (a1, t, b0), (a0, t, b0)], (0, 0.7, -0.7))
        self.quad([(x1, y, z1), (x0, y, z1), (a0, t, b1), (a1, t, b1)], (0, 0.7, 0.7))
        self.quad([(x0, y, z1), (x0, y, z0), (a0, t, b0), (a0, t, b1)], (-0.7, 0.7, 0))
        self.quad([(x1, y, z0), (x1, y, z1), (a1, t, b1), (a1, t, b0)], (0.7, 0.7, 0))
        self.quad([(a0, t, b0), (a1, t, b0), (a1, t, b1), (a0, t, b1)], (0, 1, 0))

    def gable_roof(self, x, z, w, d, y, rise):
        """人字坡顶：体育馆中央大跨度三角屋顶。"""
        x0, x1, z0, z1 = x - w / 2, x + w / 2, z - d / 2, z + d / 2
        t = y + rise
        self.quad([(x0, y, z0), (x1, y, z0), (x1, t, z), (x0, t, z)], (0, 0.7, -0.7))
        self.quad([(x1, y, z1), (x0, y, z1), (x0, t, z), (x1, t, z)], (0, 0.7, 0.7))
        self.tri([(x0, y, z0), (x0, t, z), (x0, y, z1)], (-1, 0, 0))
        self.tri([(x1, y, z1), (x1, t, z), (x1, y, z0)], (1, 0, 0))

    def dome(self, x, z, r, y, rise):
        """圆形报告厅玻璃穹顶：分层收缩圆柱近似。"""
        for i in range(4):
            f0 = 1.0 - i * 0.22
            f1 = 1.0 - (i + 1) * 0.22
            self.cylinder(x, z, r * f1, rise / 4, segments=24, base=y + rise * i / 4)

    def windows(self, x, z, w, d, h, floors):
        """逐层横向玻璃窗带，略外凸避免与墙面重叠。"""
        x0, x1, z0, z1 = x - w / 2, x + w / 2, z - d / 2, z + d / 2
        e = 0.35
        band = min(2.0, h / max(floors, 1) * 0.34)
        for i in range(1, floors + 1):
            y = h * i / (floors + 1)
            self.quad([(x0, y, z0 - e), (x1, y, z0 - e), (x1, y + band, z0 - e), (x0, y + band, z0 - e)], (0, 0, -1))
            self.quad([(x1, y, z1 + e), (x0, y, z1 + e), (x0, y + band, z1 + e), (x1, y + band, z1 + e)], (0, 0, 1))
            self.quad([(x0 - e, y, z1), (x0 - e, y, z0), (x0 - e, y + band, z0), (x0 - e, y + band, z1)], (-1, 0, 0))
            self.quad([(x1 + e, y, z0), (x1 + e, y, z1), (x1 + e, y + band, z1), (x1 + e, y + band, z0)], (1, 0, 0))

    def parapet(self, x, z, w, d, y, hh=1.2):
        x0, x1, z0, z1 = x - w / 2, x + w / 2, z - d / 2, z + d / 2
        for (cx, cz, ww, dd) in ((x, z0, w, 0.8), (x, z1, w, 0.8),
                                 (x0, z, 0.8, d), (x1, z, 0.8, d)):
            self.box(cx, cz, ww, dd, hh, base=y)


def detail_geometry(kind, x, z, w, d, h, color):
    """返回 [(Mesh, color)]，生成陶像/亭子/树列/花带等场景细节。"""
    out = []
    m = Mesh()
    if kind == "statue":
        m.cylinder(x, z, w * 0.28, h * 0.72, segments=12)
        m.cylinder(x, z, w * 0.16, h * 0.18, segments=12, base=h * 0.72)
        m.cone(x, z, w * 0.42, h * 0.10, segments=10, base=0)
        out.append((m, color))
    elif kind == "plinth":
        m.box(x, z, w, d, h)
        out.append((m, color))
    elif kind == "steps":
        for i in range(4):
            m.box(x, z + i * d * 0.06, w * (1 - i * 0.12), d * (1 - i * 0.12), h * (i + 1) / 4)
        out.append((m, STEP_COLOR))
    elif kind == "roundbed":
        m.cylinder(x, z, min(w, d) / 2, h, segments=24)
        out.append((m, color))
    elif kind == "tower":
        m.box(x, z, w, d, h)
        out.append((m, color))
        t = Mesh()
        t.cone(x, z, min(w, d) * 0.75, h * 0.28, segments=8, base=h)
        out.append((t, ROOF_COLOR))
    elif kind == "pavilion":
        m.cylinder(x, z, min(w, d) / 2, h * 0.62, segments=12)
        out.append((m, "#d8d3c6"))
        t = Mesh()
        t.cone(x, z, min(w, d) * 0.72, h * 0.48, segments=8, base=h * 0.62)
        out.append((t, ROOF_COLOR))
    elif kind == "skylight":
        m.box(x, z, w, d, h)
        out.append((m, GLASS_COLOR))
    elif kind == "bridge":
        m.box(x, z, w, d, h)
        out.append((m, color))
    elif kind == "treeline":
        n = max(6, int(d / 45))
        trunk = Mesh()
        crown = Mesh()
        for i in range(n):
            zz = z - d / 2 + d * (i + 0.5) / n
            trunk.cylinder(x, zz, 0.9, h * 0.55, segments=8)
            crown.cone(x, zz, 5.0, h * 0.75, segments=8, base=h * 0.45)
        out.append((trunk, TREE_TRUNK))
        out.append((crown, color))
    elif kind == "flowerbed":
        m.box(x, z, w, d, h)
        out.append((m, color))
    elif kind == "colonnade":
        n = max(6, int(w / 12))
        for i in range(n):
            xx = x - w / 2 + w * (i + 0.5) / n
            m.cylinder(xx, z, 0.9, h, segments=8)
        out.append((m, color))
    elif kind == "rockery":
        for i, (dx, dz, r) in enumerate(((0, 0, 1.0), (-0.3, 0.2, 0.7), (0.35, -0.2, 0.6))):
            m.cone(x + dx * w / 2, z + dz * d / 2, r * w / 5, h * (1 - i * 0.2), segments=7)
        out.append((m, color))
    elif kind == "track":
        t = 5.0
        for (cx, cz, ww, dd) in ((x, z - d / 2, w, t), (x, z + d / 2, w, t),
                                 (x - w / 2, z, t, d), (x + w / 2, z, t, d)):
            m.box(cx, cz, ww, dd, h)
        out.append((m, color))
    elif kind == "monument":
        m.box(x, z, w, d, h)
        out.append((m, color))
    else:
        m.box(x, z, w, d, h)
        out.append((m, color))
    return out


def build_parts(row):
    name, kind = row["name"], row["kind"]
    x, z = float(row["x"]), float(row["z"])
    w, d, h = float(row["w"]), float(row["d"]), float(row["h"])
    color = row["color"]
    roof = row.get("roof", "none")
    floors = int(float(row.get("floors") or 0))
    detail = row.get("detail", "none")
    parts = []

    if kind == "building":
        body = Mesh()
        if roof == "dome":
            body.cylinder(x, z, min(w, d) / 2, h, segments=24)
        else:
            body.box(x, z, w, d, h)
        parts.append((body, color))
        if floors > 0:
            g = Mesh()
            g.windows(x, z, w, d, h, floors)
            parts.append((g, GLASS_COLOR))
        r = Mesh()
        if roof == "hip":
            r.hip_roof(x, z, w, d, h, min(w, d) * 0.16)
        elif roof == "gable":
            r.gable_roof(x, z, w, d, h, min(w, d) * 0.32)
        elif roof == "dome":
            r.dome(x, z, min(w, d) / 2, h, min(w, d) * 0.22)
        elif roof == "pavilion":
            r.cone(x, z, min(w, d) * 0.72, h * 0.5, segments=8, base=h)
        elif roof == "flat":
            r.parapet(x, z, w, d, h)
        if r.pos:
            parts.append((r, ROOF_COLOR))
        for token in detail.split("+"):
            if token == "tower":
                for sgn in (-1, 1):
                    t = Mesh()
                    t.box(x + sgn * w * 0.42, z, w * 0.16, d * 0.55, h * 1.28)
                    parts.append((t, "#dfe2e4"))
                    cap = Mesh()
                    cap.cone(x + sgn * w * 0.42, z, w * 0.13, h * 0.3, segments=8, base=h * 1.28)
                    parts.append((cap, ROOF_COLOR))
            elif token == "skylight":
                s = Mesh()
                s.box(x, z, w * 0.7, d * 0.25, 1.6, base=h)
                parts.append((s, GLASS_COLOR))
            elif token == "glass":
                s = Mesh()
                s.box(x, z + d / 2 + 0.2, w * 0.86, 0.4, h * 0.62, base=h * 0.15)
                parts.append((s, GLASS_COLOR))
            elif token == "steps":
                parts.extend(detail_geometry("steps", x, z + d / 2 + 6, w * 0.5, 10, 1.6, STEP_COLOR))
            elif token == "colonnade":
                parts.extend(detail_geometry("colonnade", x, z - d / 2 - 4, w * 0.9, 2, h * 0.7, "#f2f3f0"))
    elif kind == "detail":
        parts.extend(detail_geometry(detail, x, z, w, d, h, color))
    else:
        p = Mesh()
        if "广场" in name:
            p.cylinder(x, z, min(w, d) / 2, max(h, 0.3), segments=28)
        else:
            p.plane(x, z, w, d, h)
        parts.append((p, color))
    return parts


def main():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("layout.csv 为空，终止生成")

    bin_data = bytearray()
    buffer_views, accessors, primitives, materials = [], [], [], []
    mat_index = {}
    stats, roofs, details = {}, {}, {}
    triangles = 0
    xs, zs = [], []

    def add_view(data, target):
        while len(bin_data) % 4:
            bin_data.append(0)
        off = len(bin_data)
        bin_data.extend(data)
        while len(bin_data) % 4:
            bin_data.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": off,
                             "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    for row in rows:
        parts = build_parts(row)
        for mesh, color in parts:
            if not mesh.pos:
                continue
            if color not in mat_index:
                r, g, b = hex_rgb(color)
                mat = {"name": "mat_" + color.lstrip("#"),
                       "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, 1.0],
                                                "metallicFactor": 0.0,
                                                "roughnessFactor": 0.9},
                       "doubleSided": True}
                mat_index[color] = len(materials)
                materials.append(mat)
            flat_p = [c for p in mesh.pos for c in p]
            flat_n = [c for n in mesh.nrm for c in n]
            pv = add_view(struct.pack("<%df" % len(flat_p), *flat_p), 34962)
            nv = add_view(struct.pack("<%df" % len(flat_n), *flat_n), 34962)
            iv = add_view(struct.pack("<%dH" % len(mesh.idx), *mesh.idx), 34963)
            pa = len(accessors)
            accessors.append({"bufferView": pv, "componentType": 5126,
                              "count": len(mesh.pos), "type": "VEC3",
                              "min": [min(p[i] for p in mesh.pos) for i in range(3)],
                              "max": [max(p[i] for p in mesh.pos) for i in range(3)]})
            na = len(accessors)
            accessors.append({"bufferView": nv, "componentType": 5126,
                              "count": len(mesh.nrm), "type": "VEC3"})
            ia = len(accessors)
            accessors.append({"bufferView": iv, "componentType": 5123,
                              "count": len(mesh.idx), "type": "SCALAR"})
            primitives.append({"attributes": {"POSITION": pa, "NORMAL": na},
                               "indices": ia, "material": mat_index[color], "mode": 4})
            triangles += len(mesh.idx) // 3
        k = row["kind"]
        stats[k] = stats.get(k, 0) + 1
        rf = row.get("roof", "none")
        roofs[rf] = roofs.get(rf, 0) + 1
        dt = row.get("detail", "none")
        details[dt] = details.get(dt, 0) + 1
        xs.append(float(row["x"]))
        zs.append(float(row["z"]))

    gltf = {
        "asset": {"version": "2.0", "generator": "campus-layout-builder-v2"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "方山校区布局模型（按实拍优化）"}],
        "meshes": [{"name": "campus_layout", "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_chunk = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    bin_chunk = bytes(bin_data)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = (struct.pack("<4sII", b"glTF", 2, total)
           + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
           + struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(GLB_PATH, "wb") as f:
        f.write(glb)
    report = {
        "source": "data/layout.csv + data/place_refs.csv",
        "reference": "refs/places/ 实拍照片（江苏招生考试网 2020 航拍 + 搜狐校园拼图）",
        "accuracy": "布局级示意，非测绘坐标",
        "elements": len(rows),
        "primitives": len(primitives),
        "by_kind": stats,
        "by_roof": roofs,
        "by_detail": details,
        "triangles": triangles,
        "x_range": [min(xs), max(xs)],
        "z_range": [min(zs), max(zs)],
        "glb_bytes": len(glb),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("模型已生成:", GLB_PATH, len(glb), "bytes")
    print("元素", len(rows), "| primitives", len(primitives), "| 三角面", triangles)


if __name__ == "__main__":
    main()
