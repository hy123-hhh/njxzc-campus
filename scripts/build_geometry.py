# -*- coding: utf-8 -*-
"""按官方总平面图真实轮廓挤出三维模型，并生成自包含离线预览。

仅云端 GitHub Actions 运行。输入 data/outlines.json，输出：
  models/campus_geometry.glb
  models/geometry_report.json
  docs/离线预览.html
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

import build_layout as bl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
MODEL = os.path.join(ROOT, "models", "campus_geometry.glb")
REPORT = os.path.join(ROOT, "models", "geometry_report.json")
PREVIEW = os.path.join(ROOT, "docs", "离线预览.html")
OVERLAY = os.path.join(ROOT, "docs", "outline_overlay.jpg")
PLAN_SMALL = os.path.join(ROOT, "refs", "official_01_plan.jpg")
AERIAL = os.path.join(ROOT, "refs", "places", "aerial_01.jpg")

WALL, ROOF, WATER, GREEN, ROAD = "#e7e9ea", "#3c4a63", "#6fa8cf", "#7fa85a", "#c9c8c0"


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def tris_of(points):
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            return []
        out = []
        for t in triangulate(poly):
            if poly.contains(t.representative_point()):
                coords = list(t.exterior.coords)[:3]
                out.append([(float(a), float(b)) for a, b in coords])
        return out
    except Exception:
        return []


def build_glb(polys):
    parts = []
    for p in polys:
        cls, pts = p["class"], p["points"]
        if len(pts) < 3:
            continue
        tris = tris_of(pts)
        if not tris:
            continue
        color = {"building": WALL, "water": WATER, "green": GREEN, "road": ROAD}[cls]
        if cls != "building":
            m = bl.Mesh()
            y = {"water": 0.14, "green": 0.10, "road": 0.20}[cls]
            for t in tris:
                m.tri([(a, y, b) for a, b in t], (0, 1, 0))
            parts.append((m, color))
            continue
        area = abs(Polygon(pts).area)
        h = min(34.0, max(12.0, 11.0 + math.sqrt(max(area, 1)) * 0.34))
        body = bl.Mesh()
        for t in tris:
            body.tri([(a, h, b) for a, b in t], (0, 1, 0))
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            body.quad([(a[0], 0, a[1]), (b[0], 0, b[1]),
                       (b[0], h, b[1]), (a[0], h, a[1])], (0, 0, 1))
        parts.append((body, color))
        roof = bl.Mesh()
        try:
            inner = Polygon(pts).buffer(-2.6, join_style=2)
            if not inner.is_empty and inner.geom_type == "Polygon":
                ip = list(inner.exterior.coords)[:-1]
                if len(ip) >= 3:
                    rise = min(6.0, max(2.5, h * 0.18))
                    itris = triangulate(Polygon(ip))
                    for t in itris:
                        if Polygon(ip).contains(t.representative_point()):
                            c = list(t.exterior.coords)[:3]
                            roof.tri([(a, h + rise, b) for a, b in c], (0, 1, 0))
                    n = min(len(pts), len(ip))
                    for i in range(n):
                        a, b = pts[i], pts[(i + 1) % len(pts)]
                        c, d = ip[i], ip[(i + 1) % len(ip)]
                        roof.quad([(a[0], h, a[1]), (b[0], h, b[1]),
                                   (d[0], h + rise, d[1]), (c[0], h + rise, c[1])], (0, 1, 0))
        except Exception:
            pass
        if roof.pos:
            parts.append((roof, ROOF))

    bin_data = bytearray()
    bvs, accs, prims, mats = [], [], [], []
    mat_idx = {}
    tri_count = 0

    def add_view(data, target):
        while len(bin_data) % 4:
            bin_data.append(0)
        off = len(bin_data)
        bin_data.extend(data)
        while len(bin_data) % 4:
            bin_data.append(0)
        bvs.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        return len(bvs) - 1

    for mesh, color in parts:
        if not mesh.pos:
            continue
        if color not in mat_idx:
            r, g, b = bl.hex_rgb(color)
            mat_idx[color] = len(mats)
            mats.append({"name": "mat_" + color.lstrip("#"),
                         "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, 1.0],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.88},
                         "doubleSided": True})
        fp = [c for p in mesh.pos for c in p]
        fn = [c for n in mesh.nrm for c in n]
        pv = add_view(struct.pack("<%df" % len(fp), *fp), 34962)
        nv = add_view(struct.pack("<%df" % len(fn), *fn), 34962)
        iv = add_view(struct.pack("<%dH" % len(mesh.idx), *mesh.idx), 34963)
        pa = len(accs)
        accs.append({"bufferView": pv, "componentType": 5126, "count": len(mesh.pos), "type": "VEC3",
                     "min": [min(p[i] for p in mesh.pos) for i in range(3)],
                     "max": [max(p[i] for p in mesh.pos) for i in range(3)]})
        na = len(accs)
        accs.append({"bufferView": nv, "componentType": 5126, "count": len(mesh.nrm), "type": "VEC3"})
        ia = len(accs)
        accs.append({"bufferView": iv, "componentType": 5123, "count": len(mesh.idx), "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": pa, "NORMAL": na}, "indices": ia,
                      "material": mat_idx[color], "mode": 4})
        tri_count += len(mesh.idx) // 3

    gltf = {"asset": {"version": "2.0", "generator": "campus-geometry-v1"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "按官方总平面图 1:1 轮廓挤出"}],
            "meshes": [{"name": "campus_geometry", "primitives": prims}],
            "materials": mats, "buffers": [{"byteLength": len(bin_data)}],
            "bufferViews": bvs, "accessors": accs}
    j = json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * ((4 - len(j) % 4) % 4)
    b = bytes(bin_data)
    total = 12 + 8 + len(j) + 8 + len(b)
    glb = (struct.pack("<4sII", b"glTF", 2, total)
           + struct.pack("<I4s", len(j), b"JSON") + j
           + struct.pack("<I4s", len(b), b"BIN\x00") + b)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    open(MODEL, "wb").write(glb)
    return len(glb), tri_count, len(prims)


def make_preview(polys):
    data = [{"c": p["class"], "p": p["points"], "a": p["area_px"]} for p in polys]
    imgs = {
        "plan": b64(PLAN_SMALL),
        "overlay": b64(OVERLAY) if os.path.exists(OVERLAY) else "",
        "aerial": b64(AERIAL) if os.path.exists(AERIAL) else "",
    }
    html = r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>方山校区 · 官方总平面图 1:1 轮廓模型</title>
<style>
body{margin:0;padding:22px 28px 50px;background:#fff;color:#1f2328;font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1240px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}h2{font-size:16px;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid #d8dee4}
h3{font-size:14px;font-weight:600;margin:0 0 8px}.sub,.note{color:#57606a;font-size:12.5px}
.status{background:#f6f8fa;border:1px solid #d8dee4;border-left:3px solid #1f6feb;padding:10px 14px;margin:14px 0 20px;font-size:13px}
.split{display:flex;gap:16px;flex-wrap:wrap}.pane{flex:1 1 380px;min-width:300px;border:1px solid #d8dee4;padding:10px}
.pane img{width:100%;display:block;border:1px solid #eaeef2}
canvas{width:100%;height:560px;background:#f6f8fa;border:1px solid #eaeef2;display:block;cursor:grab}
.ctl{font-size:12.5px;color:#57606a;margin-top:8px}button{border:1px solid #d8dee4;background:#f6f8fa;padding:3px 10px;cursor:pointer}
.note{background:#f6f8fa;border:1px solid #eaeef2;padding:10px 14px;margin:12px 0}
</style></head><body>
<h1>南京晓庄学院（方山校区）· 官方总平面图 1:1 轮廓模型</h1>
<p class="sub">第一阶段：几何完全复制。建筑/水体/绿地/道路的轮廓由 4604×6123 官方高清总平面图逐像素提取，非目测矩形。</p>
<div class="status">本页<b>自包含、不联网</b>。右侧三维图直接由提取出的真实轮廓挤出，形状与左上官方平面图 1:1 对应。<br>坐标为按图线性映射的示意值（非测绘成果）；材质与光影为第二阶段。</div>
<div class="split">
<div class="pane"><h3>R1 官方彩色总平面规划图（基准）</h3><img src="data:image/jpeg;base64,__PLAN__" alt="总平面图"></div>
<div class="pane"><h3>轮廓提取验收（云端自动描边）</h3><img src="data:image/jpeg;base64,__OVERLAY__" alt="轮廓叠加"></div>
</div>
<div class="split" style="margin-top:16px">
<div class="pane" style="flex:2 1 620px"><h3>1:1 轮廓三维模型（拖动旋转 · 滚轮缩放）</h3>
<canvas id="cv" width="900" height="560"></canvas>
<div class="ctl">左右拖动旋转，上下平移，滚轮缩放　<button onclick="resetView()">重置视角</button>　<label><input type="checkbox" id="showRoof" checked onchange="draw()"> 显示屋顶</label></div></div>
<div class="pane" style="flex:1 1 300px"><h3>实拍对照</h3><img src="data:image/jpeg;base64,__AERIAL__" alt="四大主楼实拍">
<p class="sub">四大主楼与中心圆形报告厅实拍。第二阶段将把实拍立面、瓦屋顶材质与光影贴到这些轮廓上。</p></div>
</div>
<div class="note">来源：学校发展规划处公开总平面规划图；实拍来自公开媒体转载，版权归原发布方。成果为教学展示用途 · 非测绘成果，不作导航或工程依据。__COUNT__ 个轮廓。</div>
<script>
const DATA=__DATA__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const C={building:['#e7e9ea','#3c4a63'],water:'#6fa8cf',green:'#7fa85a',road:'#c9c8c0'};
let ang=-.62,sc=.48,panX=0,panY=110,drag=false,lx=0,ly=0;
function sh(h,f){const s=h.replace('#','');return 'rgb('+[0,2,4].map(i=>Math.min(255,parseInt(s.slice(i,i+2),16)*f|0)).join(',')+')';}
function proj(x,y,z){const c=Math.cos(ang),s=Math.sin(ang),rx=x*c-z*s,rz=x*s+z*c;
return [cv.width/2+(rx-rz)*.866*sc+panX,cv.height/2+(rx+rz)*.5*sc-y*sc*.9+panY];}
function poly3(pts,fill){ctx.beginPath();pts.forEach((p,i)=>{const q=proj(p[0],p[1],p[2]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle='rgba(40,50,65,.28)';ctx.lineWidth=.6;ctx.stroke();}
function area(p){let a=0;for(let i=0;i<p.length;i++){const q=p[(i+1)%p.length];a+=p[i][0]*q[1]-q[0]*p[i][1];}return Math.abs(a)/2;}
function height(a){return Math.min(34,Math.max(12,11+Math.sqrt(Math.max(a,1))*.34));}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);const roof=document.getElementById('showRoof').checked;
const order=DATA.slice().sort((a,b)=>{const ca=a.p.reduce((s,p)=>s+p[0]+p[1],0)/a.p.length,cb=b.p.reduce((s,p)=>s+p[0]+p[1],0)/b.p.length;return ca-cb;});
order.forEach(r=>{const p=r.p;if(p.length<3)return;
if(r.c!=='building'){const y=r.c==='water'?.14:(r.c==='green'?.1:.2);poly3(p.map(q=>[q[0],y,q[1]]),C[r.c]);return;}
const a=area(p),h=height(a);
const cx=p.reduce((s,q)=>s+q[0],0)/p.length,cz=p.reduce((s,q)=>s+q[1],0)/p.length;
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length];
poly3([[u[0],0,u[1]],[v[0],0,v[1]],[v[0],h,v[1]],[u[0],h,u[1]]],sh(C.building[0],.72+.28*((i%3)/3)));}
poly3(p.map(q=>[q[0],h,q[1]]),sh(C.building[0],1.06));
if(roof){const k=.12,rise=Math.min(6,Math.max(2.5,h*.18));
const ip=p.map(q=>[cx+(q[0]-cx)*(1-k),cz+(q[1]-cz)*(1-k)]);
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length],c=ip[i],d=ip[(i+1)%ip.length];
poly3([[u[0],h,u[1]],[v[0],h,v[1]],[d[0],h+rise,d[1]],[c[0],h+rise,c[1]]],sh(C.building[1],.8+.25*((i%3)/3)));}
poly3(ip.map(q=>[q[0],h+rise,q[1]]),sh(C.building[1],1.14));}
});}
function resetView(){ang=-.62;sc=.48;panX=0;panY=110;draw();}
cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=false;cv.style.cursor='grab';});
window.addEventListener('mousemove',e=>{if(!drag)return;ang+=(e.clientX-lx)*.008;panY+=(e.clientY-ly)*.5;lx=e.clientX;ly=e.clientY;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();sc*=e.deltaY>0?.9:1.1;draw();},{passive:false});
draw();</script></body></html>'''
    html = (html.replace("__PLAN__", imgs["plan"])
                .replace("__OVERLAY__", imgs["overlay"])
                .replace("__AERIAL__", imgs["aerial"])
                .replace("__COUNT__", str(len(polys)))
                .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    open(PREVIEW, "w", encoding="utf-8", newline="\n").write(html)
    return len(html.encode("utf-8"))


def main():
    d = json.load(open(OUTLINES, encoding="utf-8"))
    polys = d["polygons"]
    size, tris, prims = build_glb(polys)
    html_size = make_preview(polys)
    rep = {"source": "refs/official_01_plan_d.jpg 逐像素轮廓提取",
           "accuracy": "形状与官方总平面图 1:1；坐标为线性映射示意值，非测绘成果",
           "polygons": len(polys), "primitives": prims, "triangles": tris,
           "glb_bytes": size, "preview_bytes": html_size,
           "by_class": {c: sum(1 for p in polys if p["class"] == c)
                        for c in ("building", "water", "green", "road")}}
    open(REPORT, "w", encoding="utf-8").write(json.dumps(rep, ensure_ascii=False, indent=2))
    print("几何模型:", size, "bytes | 轮廓", len(polys), "| 三角面", tris)
    print("离线预览:", html_size, "bytes")


if __name__ == "__main__":
    main()
