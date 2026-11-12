# -*- coding: utf-8 -*-
"""离线校验渲染：严格复刻 离线预览.html 的 draw3D 投影与徽派上色（含地基底板）。"""
import json, math, os
import io
import numpy as np
from PIL import Image, ImageDraw

from build_materials import load_ground  # 复用沙盘校准后的地面贴图（擦图例/草坪绿）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
GROUND = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_verify_render.png")

d = json.load(open(OUTLINES, encoding="utf-8"))
polys = d["polygons"]

W, H = 1000, 600
ANG = -0.62
PANX, PANY = 0, 140
allp = [(q[0], q[1]) for p in polys for q in p["points"]]
xs = [p[0] for p in allp]; zs = [p[1] for p in allp]
span = max(max(xs) - min(xs), max(zs) - min(zs), 1.0)
SC = round(336.0 / span, 5)
C, S = math.cos(ANG), math.sin(ANG)

def proj(x, y, z):
    rx = x * C - z * S
    rz = x * S + z * C
    return [W / 2 + (rx - rz) * 0.866 * SC + PANX,
            H / 2 + (rx + rz) * 0.5 * SC - y * SC * 0.9 + PANY]

def P(x, y, z):
    return (int(proj(x, y, z)[0]), int(proj(x, y, z)[1]))

def shade(rgb, f):
    return tuple(min(255, max(0, int(c * f))) for c in rgb)

def area(p):
    a = 0
    for i in range(len(p)):
        q = p[(i + 1) % len(p)]
        a += p[i][0] * q[1] - q[0] * p[i][1]
    return abs(a) / 2

def height(a):
    return 21 if a > 9000 else (18 if a > 3500 else (15 if a > 1200 else 11))

def long_edges(p):
    if len(p) < 3: return []
    best, bA, bB = 0.0, p[0], p[1]
    for i in range(len(p)):
        a2 = p[i]; b2 = p[(i + 1) % len(p)]
        dd = math.hypot(b2[0]-a2[0], b2[1]-a2[1])
        if dd > best: best, bA, bB = dd, a2, b2
    ang0 = math.atan2(bB[1]-bA[1], bB[0]-bA[0])
    edges = []
    for i in range(len(p)):
        a2 = p[i]; b2 = p[(i + 1) % len(p)]
        if math.hypot(b2[0]-a2[0], b2[1]-a2[1]) < 1: continue
        da = math.atan2(b2[1]-a2[1], b2[0]-a2[0]) - ang0
        da = abs(math.atan2(math.sin(da), math.cos(da)))
        if da < 0.35: edges.append((a2, b2))
    return edges

def obb(p):
    n = len(p); cx = sum(q[0] for q in p)/n; cz = sum(q[1] for q in p)/n
    best, bA, bB = 0.0, p[0], p[1]
    for i in range(n):
        a2 = p[i]; b2 = p[(i+1)%n]; dd = math.hypot(b2[0]-a2[0], b2[1]-a2[1])
        if dd > best: best, bA, bB = dd, a2, b2
    ux, uz = (bB[0]-bA[0])/best, (bB[1]-bA[1])/best; vx, vz = -uz, ux
    Lh = Wh = 0.0
    for q in p:
        Lh = max(Lh, abs((q[0]-cx)*ux+(q[1]-cz)*uz)); Wh = max(Wh, abs((q[0]-cx)*vx+(q[1]-cz)*vz))
    return cx, cz, ux, uz, vx, vz, Lh, Wh

COL = {"building": (224, 226, 225), "roof": (108, 114, 120), "maotou": (34, 39, 44),
       "water": (111, 168, 207), "green": (127, 168, 90), "road": (192, 196, 188)}

# 背景：暖灰绿地平
img = Image.new("RGB", (W, H), (233, 236, 230))
px = img.load()
for y in range(H):
    t = y / H
    if t < 0.55:
        f = t / 0.55; r = int(233+(238-233)*f); g = int(236+(241-236)*f); b = int(230+(234-230)*f)
    else:
        f = (t-0.55)/0.45; r = int(238+(227-238)*f); g = int(241+(230-241)*f); b = int(234+(220-234)*f)
    for x in range(W): px[x, y] = (r, g, b)
draw = ImageDraw.Draw(img, "RGBA")

# 地面：规划图仿射贴图（dest像素 -> source图块像素，须解逆映射）
gw, gh = 1800, 2395
def wx(u): return (u-900)/1.81
def wz(v): return (v-1197.5)/1.81
src = np.array([(0, 0), (gw, 0), (0, gh)], float)          # 规划图像素
dst = np.array([proj(wx(u), 0, wz(v)) for u, v in src], float)  # 屏幕像素
A = np.column_stack([dst, np.ones(3)])                     # [dx, dy, 1]
coef_u, *_ = np.linalg.lstsq(A, src[:, 0], rcond=None)     # u = a*dx+b*dy+c
coef_v, *_ = np.linalg.lstsq(A, src[:, 1], rcond=None)     # v = d*dx+e*dy+f
a, b, c = coef_u; dd, e, f = coef_v
gimg = Image.open(io.BytesIO(load_ground())).convert("RGB").resize((gw, gh))
ground = gimg.transform((W, H), Image.AFFINE, (a, b, c, dd, e, f), resample=Image.BILINEAR)

# 地基底板（先画侧立面；顶面随后由规划图贴图覆盖）
def draw_base():
    X0,Z0,X1,Z1=-505,-668,505,668
    front=[P(X0,0,Z1),P(X1,0,Z1),P(X1,-0.6,Z1),P(X0,-0.6,Z1)]
    right=[P(X1,0,Z1),P(X1,0,Z0),P(X1,-0.6,Z0),P(X1,-0.6,Z1)]
    draw.polygon(front, fill=(169,167,155))
    draw.polygon(right, fill=(182,180,168))
draw_base()

# 规划图贴到顶面（只贴校园四边形范围内）
quad = [P(wx(0),0,wz(0)), P(wx(gw),0,wz(0)), P(wx(gw),0,wz(gh)), P(wx(0),0,wz(gh))]
mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask).polygon([tuple(q) for q in quad], fill=255)
img = Image.composite(ground, img, mask)
draw = ImageDraw.Draw(img, "RGBA")

def poly(pts2d, y, fill, outline=None):
    sp = [P(x, y, z) for x, z in pts2d]
    draw.polygon(sp, fill=fill, outline=outline)

# 水/绿/路/操场由规划图贴图地面承载（与 GLB 一致），不再铺平色覆盖层

blds = [p for p in polys if p["class"] == "building"]
blds.sort(key=lambda p: sum(q[0]+q[1] for q in p["points"])/len(p["points"]))

def pip(px, pz, poly):
    inside = False; N = len(poly); j = N - 1
    for i2 in range(N):
        xi, zi = poly[i2]; xj, zj = poly[j]
        if (zi > pz) != (zj > pz) and px < (xj - xi) * (pz - zi) / (zj - zi + 1e-12) + xi:
            inside = not inside
        j = i2
    return inside

for p in blds:
    pts = p["points"]
    if len(pts) < 3: continue
    h = height(area(pts))
    cx, cz, ux, uz, vx, vz, Lh, Wh = obb(pts)
    for i in range(len(pts)):
        u = pts[i]; v = pts[(i+1)%len(pts)]
        ang_e = math.atan2(v[1]-u[1], v[0]-u[0])
        lit = 0.72 + 0.28*abs(math.cos(ang_e+ANG))
        q = [P(u[0],0,u[1]), P(v[0],0,v[1]), P(v[0],h,v[1]), P(u[0],h,u[1])]
        draw.polygon(q, fill=shade(COL["building"], lit))
    k = 0.12; rise = min(2.6, max(1.8, h*0.15))
    ip = [[cx+(q[0]-cx)*(1-k), cz+(q[1]-cz)*(1-k)] for q in pts]
    for i in range(len(pts)):
        u = pts[i]; v = pts[(i+1)%len(pts)]; c2 = ip[i]; d2 = ip[(i+1)%len(ip)]
        q = [P(u[0],h,u[1]), P(v[0],h,v[1]), P(d2[0],h+rise,d2[1]), P(c2[0],h+rise,c2[1])]
        draw.polygon(q, fill=shade(COL["roof"], 0.82+0.2*((i%3)/3)))
    qtop = [P(q[0],h+rise,q[1]) for q in ip]
    draw.polygon(qtop, fill=shade(COL["roof"], 1.15))
    draw.line(qtop+[qtop[0]], fill=(20,26,30,115), width=1)
    qeave = [P(q[0],h,q[1]) for q in pts]
    draw.line(qeave+[qeave[0]], fill=(30,38,46,90), width=1)
    # 马头墙：① 檐口压顶（每条边）
    for i in range(len(pts)):
        A=pts[i]; B=pts[(i+1)%len(pts)]
        dx=B[0]-A[0]; dy=B[1]-A[1]; L=math.hypot(dx,dy)
        if L<2: continue
        bot=[]; top=[]
        for j in range(5):
            t=j/4; x=A[0]+dx*t; z=A[1]+dy*t; bot.append(P(x,h,z)); top.append(P(x,h+1.0,z))
        draw.polygon(bot+top[::-1], fill=COL["maotou"], outline=(16,20,24))
    # ② 近矩形：长轴两端山墙阶梯（马头墙）
    n=len(pts); cx2=sum(q[0] for q in pts)/n; cz2=sum(q[1] for q in pts)/n
    best,bA,bB=0.0,pts[0],pts[1]
    for i in range(n):
        a=pts[i]; b=pts[(i+1)%n]; dd=math.hypot(b[0]-a[0], b[1]-a[1])
        if dd>best: best,bA,bB=dd,a,b
    ux,uz=(bB[0]-bA[0])/best,(bB[1]-bA[1])/best; vx,vz=-uz,ux
    Lh=Wh=0.0
    for q in pts: Lh=max(Lh,abs((q[0]-cx2)*ux+(q[1]-cz2)*uz)); Wh=max(Wh,abs((q[0]-cx2)*vx+(q[1]-cz2)*vz))
    ex1=(cx2+Lh*ux,cz2+Lh*uz); ex2=(cx2-Lh*ux,cz2-Lh*uz)
    near=min(min(math.hypot(ex1[0]-q[0],ex1[1]-q[1]) for q in pts),min(math.hypot(ex2[0]-q[0],ex2[1]-q[1]) for q in pts))
    # ② 长轴两端阶梯山墙：先找端面实际覆盖区段，再逐盒包含复查
    us = []; vs = []
    for q in pts:
        us.append((q[0]-cx2)*ux+(q[1]-cz2)*uz); vs.append((q[0]-cx2)*vx+(q[1]-cz2)*vz)
    umin, umax = min(us), max(us); vmin, vmax = min(vs), max(vs)
    M=5; base=1.2; step=1.6; thick=1.4; SAMP=80
    for u_end, e in ((umax, 1), (umin, -1)):
        ui=u_end-e*thick*0.5
        flags=[pip(cx2+ui*ux+(vmin+(vmax-vmin)*s/SAMP)*vx, cz2+ui*uz+(vmin+(vmax-vmin)*s/SAMP)*vz, pts) for s in range(SAMP+1)]
        bs=bl=cs=cl=0
        for s, f in enumerate(flags+[False]):
            if f:
                if cl==0: cs=s
                cl+=1
            else:
                if cl>bl: bs,bl=cs,cl
                cl=0
        if bl<4 or bl/SAMP*(vmax-vmin)<6.0: continue
        v_lo=vmin+(vmax-vmin)*bs/SAMP; v_hi=vmin+(vmax-vmin)*(bs+bl)/SAMP
        vseg=(v_hi-v_lo)/M
        for i in range(M):
            vc=v_lo+(i+0.5)*vseg; lvl=base+step*(2-abs(i-2))
            ok=True
            for t5 in range(5):
                vv=(vc-vseg*0.49)+(vseg*0.98)*t5/4
                if not pip(cx2+ui*ux+vv*vx, cz2+ui*uz+vv*vz, pts): ok=False; break
            if not ok: continue
            ccx=cx2+u_end*ux+vc*vx; ccz=cz2+u_end*uz+vc*vz; hu=thick*0.5; hv=vseg*0.49
            cs2=[[-hu,-hv],[hu,-hv],[hu,hv],[-hu,hv]]
            o=[P(ccx+s2[0]*ux+s2[1]*vx, h, ccz+s2[0]*uz+s2[1]*vz) for s2 in cs2]
            t=[P(ccx+s2[0]*ux+s2[1]*vx, h+lvl, ccz+s2[0]*uz+s2[1]*vz) for s2 in cs2]
            draw.polygon(o+[t[3],t[2],t[1],t[0]], fill=COL["maotou"], outline=(16,20,24))
    for (A,B) in long_edges(pts):
        dx=B[0]-A[0]; dy=B[1]-A[1]; L=math.hypot(dx,dy)
        if L<2: continue
        bot=[]; top=[]
        for i in range(5):
            t=i/4; x=A[0]+dx*t; z=A[1]+dy*t
            bot.append(P(x,h,z)); top.append(P(x,h+0.5,z))
        draw.polygon(bot+top[::-1], fill=COL["maotou"], outline=(16,20,24))

img.save(OUT)
print("saved", OUT)
