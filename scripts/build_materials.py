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

TEX = {
    "roof": (60, 74, 99), "wall": (231, 233, 234),
    "water": (111, 168, 207), "green": (127, 168, 90),
    "road": (201, 200, 192), "glass": (143, 179, 206),
}


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


def build_parts(polys):
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
        h = min(34.0, max(12.0, 11.0 + math.sqrt(max(area, 1)) * 0.34))
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
        parts.append((body, "wall"))
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
    return parts


def build_glb(parts, textures):
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

    for name, raw in textures.items():
        imgs.append({"bufferView": add_view(raw), "mimeType": "image/png", "name": name})
    tex_objs = [{"sampler": 0, "source": i} for i in range(len(imgs))]
    mat_index = {}

    for mesh, key in parts:
        if not mesh.pos:
            continue
        if key not in mat_index:
            r, g, b = [c / 255.0 for c in TEX.get(key, (200, 200, 200))[::-1]]
            mat_index[key] = len(mats)
            mats.append({"name": key, "doubleSided": True,
                         "pbrMetallicRoughness": {
                             "baseColorTexture": {"index": list(textures).index(key)},
                             "baseColorFactor": [1, 1, 1, 1],
                             "metallicFactor": 0.05 if key != "water" else 0.25,
                             "roughnessFactor": 0.45 if key in ("water", "glass") else 0.85}})
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
                      "indices": ia, "material": mat_index[key], "mode": 4})
        tri_count += len(mesh.idx) // 3

    gltf = {"asset": {"version": "2.0", "generator": "campus-materials-v1"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "方山校区 1:1 轮廓 + 程序化材质"}],
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


def make_preview(polys, textures):
    data = [{"c": p["class"], "p": p["points"]} for p in polys]
    tex_b64 = {k: base64.b64encode(v).decode() for k, v in textures.items()}
    imgs = {"plan": b64(os.path.join(ROOT, "refs", "official_01_plan.jpg")),
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
<h1>南京晓庄学院（方山校区）· 1:1 轮廓 + 材质光影</h1>
<p class="sub">第二阶段：几何形状逐像素取自官方总平面图，材质按实拍配色程序化生成（瓦屋顶 / 窗格墙面 / 水波 / 草地 / 铺装），含光照与投影。</p>
<div class="status">本页<b>自包含、不联网</b>。下方三张参考图与 6 张材质纹理全部内嵌。<br>材质为程序化近似（按实拍配色），非照片贴图；坐标为线性映射示意值，非测绘成果。</div>
<div class="split">
<div class="pane"><h3>R1 官方总平面规划图（几何基准）</h3><img src="data:image/jpeg;base64,__PLAN__" alt="总平面图"></div>
<div class="pane"><h3>云端描边验收</h3><img src="data:image/jpeg;base64,__OVERLAY__" alt="描边验收"></div>
<div class="pane"><h3>四大主楼实拍（配色依据）</h3><img src="data:image/jpeg;base64,__AERIAL__" alt="实拍"></div>
</div>
<h2>材质光影三维预览</h2>
<canvas id="cv" width="1000" height="580"></canvas>
<div class="ctl">左右拖动旋转，上下平移，滚轮缩放　<button onclick="resetView()">重置视角</button>
<label><input type="checkbox" id="shadow" checked onchange="draw()"> 投影</label>
<label><input type="checkbox" id="texture" checked onchange="draw()"> 材质纹理</label>
<label><input type="checkbox" id="tree" checked onchange="draw()"> 树木</label></div>
<div class="note">共 __COUNT__ 个真实轮廓。来源：学校发展规划处公开总平面规划图；实拍来自公开媒体转载，版权归原发布方。教学展示用途 · 示意性布局 · 非测绘成果。</div>
<script>
const DATA=__DATA__;
const TEX={__TEX__};
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const PAT={};
Object.keys(TEX).forEach(k=>{const im=new Image();im.onload=()=>{PAT[k]=ctx.createPattern(im,'repeat');draw();};im.src='data:image/png;base64,'+TEX[k];});
const BASE={building:'#e7e9ea',water:'#6fa8cf',green:'#7fa85a',road:'#c9c8c0',roof:'#3c4a63'};
let ang=-.62,sc=.48,panX=0,panY=120,drag=false,lx=0,ly=0;
function sh(h,f){const s=h.replace('#','');return 'rgb('+[0,2,4].map(i=>Math.min(255,parseInt(s.slice(i,i+2),16)*f|0)).join(',')+')';}
function proj(x,y,z){const c=Math.cos(ang),s=Math.sin(ang),rx=x*c-z*s,rz=x*s+z*c;
return [cv.width/2+(rx-rz)*.866*sc+panX,cv.height/2+(rx+rz)*.5*sc-y*sc*.9+panY];}
function poly3(pts,fill,mode){ctx.beginPath();pts.forEach((p,i)=>{const q=proj(p[0],p[1],p[2]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
ctx.closePath();
const texOn=document.getElementById('texture').checked;
if(mode&&texOn&&PAT[mode]){ctx.save();ctx.fillStyle=PAT[mode];ctx.globalAlpha=1;ctx.fill();ctx.restore();}
else{ctx.fillStyle=fill;ctx.fill();}
ctx.strokeStyle='rgba(35,45,60,.30)';ctx.lineWidth=.5;ctx.stroke();}
function area(p){let a=0;for(let i=0;i<p.length;i++){const q=p[(i+1)%p.length];a+=p[i][0]*q[1]-q[0]*p[i][1];}return Math.abs(a)/2;}
function height(a){return Math.min(34,Math.max(12,11+Math.sqrt(Math.max(a,1))*.34));}
function fillBg(){const g=ctx.createLinearGradient(0,0,0,cv.height);g.addColorStop(0,'#dfeaf5');g.addColorStop(.55,'#eef3f7');g.addColorStop(1,'#e7eddf');ctx.fillStyle=g;ctx.fillRect(0,0,cv.width,cv.height);}
function draw(){fillBg();const shadow=document.getElementById('shadow').checked,trees=document.getElementById('tree').checked;
const order=DATA.slice().sort((a,b)=>{const ca=a.p.reduce((s,p)=>s+p[0]+p[1],0)/a.p.length,cb=b.p.reduce((s,p)=>s+p[0]+p[1],0)/b.p.length;return ca-cb;});
order.forEach(r=>{const p=r.p;if(p.length<3)return;
if(r.c!=='building'){const y=r.c==='water'?.14:(r.c==='green'?.1:.2);
poly3(p.map(q=>[q[0],y,q[1]]),BASE[r.c],r.c==='green'?'green':(r.c==='water'?'water':'road'));
if(r.c==='green'&&trees){const cx=p.reduce((s,q)=>s+q[0],0)/p.length,cz=p.reduce((s,q)=>s+q[1],0)/p.length;
for(let i=0;i<Math.min(10,Math.max(2,Math.floor(area(p)/900)));i++){const a=i*2.4;const q=proj(cx+Math.cos(a)*9,3,cz+Math.sin(a)*9);
ctx.beginPath();ctx.arc(q[0],q[1],3.2,0,7);ctx.fillStyle='rgba(72,116,62,.75)';ctx.fill();}}
return;}
const a=area(p),h=height(a),cx=p.reduce((s,q)=>s+q[0],0)/p.length,cz=p.reduce((s,q)=>s+q[1],0)/p.length;
if(shadow){poly3(p.map(q=>[q[0]+14,.02,q[1]+10]),'rgba(52,64,58,.16)');}
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length];
const lit=.70+.30*Math.abs(Math.cos(Math.atan2(v[1]-u[1],v[0]-u[0])+ang));
poly3([[u[0],0,u[1]],[v[0],0,v[1]],[v[0],h,v[1]],[u[0],h,u[1]]],sh(BASE.building,lit),'wall');}
poly3(p.map(q=>[q[0],h,q[1]]),sh(BASE.building,1.05),'wall');
const k=.12,rise=Math.min(6,Math.max(2.5,h*.18)),ip=p.map(q=>[cx+(q[0]-cx)*(1-k),cz+(q[1]-cz)*(1-k)]);
for(let i=0;i<p.length;i++){const u=p[i],v=p[(i+1)%p.length],c=ip[i],d=ip[(i+1)%ip.length];
poly3([[u[0],h,u[1]],[v[0],h,v[1]],[d[0],h+rise,d[1]],[c[0],h+rise,c[1]]],sh(BASE.roof,.82+.2*((i%3)/3)),'roof');}
poly3(ip.map(q=>[q[0],h+rise,q[1]]),sh(BASE.roof,1.15),'roof');
});}
function resetView(){ang=-.62;sc=.48;panX=0;panY=120;draw();}
cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{drag=false;cv.style.cursor='grab';});
window.addEventListener('mousemove',e=>{if(!drag)return;ang+=(e.clientX-lx)*.008;panY+=(e.clientY-ly)*.5;lx=e.clientX;ly=e.clientY;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();sc*=e.deltaY>0?.9:1.1;draw();},{passive:false});
draw();</script></body></html>'''
    tex_js = ",".join("'%s':'%s'" % (k, v) for k, v in tex_b64.items())
    html = (html.replace("__PLAN__", imgs["plan"]).replace("__OVERLAY__", imgs["overlay"])
                .replace("__AERIAL__", imgs["aerial"]).replace("__COUNT__", str(len(polys)))
                .replace("__TEX__", tex_js).replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    open(PREVIEW, "w", encoding="utf-8", newline="\n").write(html)
    return len(html.encode("utf-8"))


def main():
    d = json.load(open(OUTLINES, encoding="utf-8"))
    polys = d["polygons"]
    textures = make_textures()
    parts = build_parts(polys)
    size, tris, prims = build_glb(parts, textures)
    html_size = make_preview(polys, textures)
    rep = {"stage": "材质与光影", "source": "官方总平面图真实轮廓 + 程序化材质（按实拍配色）",
           "polygons": len(polys), "primitives": prims, "triangles": tris,
           "textures": list(textures), "glb_bytes": size, "preview_bytes": html_size,
           "note": "材质为程序化近似，非逐栋照片贴图；需要每栋正面照才能进一步照片级还原"}
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("材质模型:", size, "bytes | 三角面", tris, "| 纹理", len(textures))
    print("离线预览:", html_size, "bytes")


if __name__ == "__main__":
    main()
