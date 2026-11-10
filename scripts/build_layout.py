# -*- coding: utf-8 -*-
"""方山校区布局级白模生成器（纯 Python 标准库，零第三方依赖）。

从 data/layout.csv 读取布局元素，生成 glTF 2.0 二进制 GLB。
仅供 GitHub Actions 云端运行；本地不执行建模任务。

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

    def box(self, x, z, w, d, h):
        x0, x1 = x - w / 2, x + w / 2
        z0, z1 = z - d / 2, z + d / 2
        y0, y1 = 0.0, h
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

    def cylinder(self, x, z, radius, h, segments=24):
        y0, y1 = 0.0, h
        for i in range(segments):
            a0 = 2 * math.pi * i / segments
            a1 = 2 * math.pi * (i + 1) / segments
            p0 = (x + radius * math.cos(a0), y1, z + radius * math.sin(a0))
            p1 = (x + radius * math.cos(a1), y1, z + radius * math.sin(a1))
            self.tri([(x, y1, z), p0, p1], (0, 1, 0))
            q0 = (p0[0], y0, p0[2])
            q1 = (p1[0], y0, p1[2])
            nx = math.cos((a0 + a1) / 2)
            nz = math.sin((a0 + a1) / 2)
            self.quad([q0, p0, p1, q1], (nx, 0, nz))


def build_mesh(row):
    name = row["name"]
    kind = row["kind"]
    x, z = float(row["x"]), float(row["z"])
    w, d, h = float(row["w"]), float(row["d"]), float(row["h"])
    m = Mesh()
    if kind == "building":
        if "碟形" in name:
            m.cylinder(x, z, min(w, d) / 2, h)
        else:
            m.box(x, z, w, d, h)
    elif "广场" in name:
        m.cylinder(x, z, min(w, d) / 2, max(h, 0.3))
    else:
        m.plane(x, z, w, d, h)
    return m


def main():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("layout.csv 为空，终止生成（避免产出与参考图不一致的模型）")

    bin_data = bytearray()
    buffer_views = []
    accessors = []
    primitives = []
    materials = []
    mat_index = {}
    stats = {}
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
        color = row["color"].strip()
        if color not in mat_index:
            r, g, b = hex_rgb(color)
            mat = {
                "name": "mat_" + color.lstrip("#"),
                "pbrMetallicRoughness": {
                    "baseColorFactor": [r, g, b, 0.88 if row["kind"] == "water" else 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
                "doubleSided": True,
            }
            if row["kind"] == "water":
                mat["alphaMode"] = "BLEND"
            mat_index[color] = len(materials)
            materials.append(mat)

        mesh = build_mesh(row)
        flat_p = [c for p in mesh.pos for c in p]
        flat_n = [c for n in mesh.nrm for c in n]
        if not flat_p:
            continue
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
        xs.append(float(row["x"]))
        zs.append(float(row["z"]))
        k = row["kind"]
        stats[k] = stats.get(k, 0) + 1

    gltf = {
        "asset": {"version": "2.0", "generator": "campus-layout-builder"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "方山校区布局级白模"}],
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
        "source": "data/layout.csv",
        "reference": "refs/official_01_plan.jpg（R1 彩色总平面规划图）",
        "accuracy": "布局级示意，非测绘坐标",
        "elements": len(rows),
        "by_kind": stats,
        "triangles": triangles,
        "x_range": [min(xs), max(xs)],
        "z_range": [min(zs), max(zs)],
        "glb_bytes": len(glb),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("模型已生成:", GLB_PATH, len(glb), "bytes")
    print("元素", len(rows), "个，三角面", triangles, "个")


if __name__ == "__main__":
    main()
