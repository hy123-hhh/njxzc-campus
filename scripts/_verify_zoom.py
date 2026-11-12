# -*- coding: utf-8 -*-
"""放大校验：把单栋楼拉近渲染，确认 粉墙/黛瓦/马头墙 在可辨识尺度下是否成立。"""
import json, math, os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_verify_zoom.png")
d = json.load(open(OUTLINES, encoding="utf-8"))
polys = d["polygons"]

W, H = 1000, 600
ANG = -1.05
C, S = math.cos(ANG), math.sin(ANG)
allb = [p for p in polys if p["class"] == "building"]
def area(p):
    a = 0
    for i in range(len(p)):
        q = p[(i+1)%len(p)]; a += p[i][0]*q[1]-q[0]*p[i][1]
    return abs(a)/2
def hgt(a): return 21 if a>9000 else (18 if a>3500 else (15 if a>1200 else 11))
def near_val(pts):
    n=len(pts); cx=sum(q[0] for q in pts)/n; cz=sum(q[1] for q in pts)/n
    best,bA,bB=0.0,pts[0],pts[1]
    for i in range(n):
        a=pts[i]; b=pts[(i+1)%n]; dd=math.hypot(b[0]-a[0],b[1]-a[1])
        if dd>best: best,bA,bB=dd,a,b
    ux,uz=(bB[0]-bA[0])/best,(bB[1]-bA[1])/best; vx,vz=-uz,ux
    Lh=Wh=0.0
    for q in pts: Lh=max(Lh,abs((q[0]-cx)*ux+(q[1]-cz)*uz)); Wh=max(Wh,abs((q[0]-cx)*vx+(q[1]-cz)*vz))
    ex1=(cx+Lh*ux,cz+Lh*uz); ex2=(cx-Lh*ux,cz-Lh*uz)
    near=min(min(math.hypot(ex1[0]-q[0],ex1[1]-q[1]) for q in pts),min(math.hypot(ex2[0]-q[0],ex2[1]-q[1]) for q in pts))
    return near, Lh, Wh
# 选一栋「最像矩形且最长」的楼：必触发马头墙、且呈现经典长墙两端阶梯山墙轮廓
cand=[]
for p in allb:
    pts=p["points"]; a=area(pts); h=hgt(a)
    if len(pts)<4: continue
    nr,Lh,Wh=near_val(pts)
    if nr<=max(5.0,0.2*Lh):
        cand.append((Lh, nr, Wh, a, h, pts))
cand.sort(reverse=True)
_, near, Wh, a0, h, pts = cand[0]
Lh = cand[0][0]
cx = sum(q[0] for q in pts)/len(pts); cz = sum(q[1] for q in pts)/len(pts)
SC = min((W*0.8)/(Lh+Wh+8), (H*0.8)/(h+10))
yc = h*0.5
rx=cx*C-cz*S; rz=cx*S+cz*C
PANX = -(rx-rz)*0.866*SC
PANY = -((rx+rz)*0.5*SC - yc*SC*1.6)

def proj(x,y,z):
    rrx=x*C-z*S; rrz=x*S+z*C
    return [W/2+(rrx-rrz)*0.866*SC+PANX, H/2+(rrx+rrz)*0.5*SC-y*SC*1.6+PANY]
def P(x,y,z): return (int(proj(x,y,z)[0]), int(proj(x,y,z)[1]))
def shade(rgb,f): return tuple(min(255,max(0,int(c*f))) for c in rgb)

def long_edges(p):
    if len(p)<3: return []
    best,bA,bB=0.0,p[0],p[1]
    for i in range(len(p)):
        a2=p[i];b2=p[(i+1)%len(p)];dd=math.hypot(b2[0]-a2[0],b2[1]-a2[1])
        if dd>best:best,bA,bB=dd,a2,b2
    ang0=math.atan2(bB[1]-bA[1],bB[0]-bA[0]);edges=[]
    for i in range(len(p)):
        a2=p[i];b2=p[(i+1)%len(p)]
        if math.hypot(b2[0]-a2[0],b2[1]-a2[1])<1:continue
        da=math.atan2(b2[1]-a2[1],b2[0]-a2[0])-ang0;da=abs(math.atan2(math.sin(da),math.cos(da)))
        if da<0.35:edges.append((a2,b2))
    return edges

def obb(p):
    n=len(p);cx2=sum(q[0] for q in p)/n;cz2=sum(q[1] for q in p)/n
    best,bA,bB=0.0,p[0],p[1]
    for i in range(n):
        a2=p[i];b2=p[(i+1)%n];dd=math.hypot(b2[0]-a2[0],b2[1]-a2[1])
        if dd>best:best,bA,bB=dd,a2,b2
    ux,uz=(bB[0]-bA[0])/best,(bB[1]-bA[1])/best;vx,vz=-uz,ux
    Lh=Wh=0.0
    for q in p:Lh=max(Lh,abs((q[0]-cx2)*ux+(q[1]-cz2)*uz));Wh=max(Wh,abs((q[0]-cx2)*vx+(q[1]-cz2)*vz))
    return cx2,cz2,ux,uz,vx,vz,Lh,Wh

COL={"building":(224,226,225),"roof":(108,114,120),"maotou":(34,39,44)}

img=Image.new("RGB",(W,H),(233,236,230));draw=ImageDraw.Draw(img,"RGBA")
# 地面小板
gx0,gz0,gx1,gz1=cx-45,cz-45,cx+45,cz+45
draw.polygon([P(gx0,0,gz0),P(gx1,0,gz0),P(gx1,0,gz1),P(gx0,0,gz1)],fill=(201,199,187))
draw.polygon([P(gx0,0,gz1),P(gx1,0,gz1),P(gx1,-0.6,gz1),P(gx0,-0.6,gz1)],fill=(169,167,155))
draw.polygon([P(gx1,0,gz1),P(gx1,0,gz0),P(gx1,-0.6,gz0),P(gx1,-0.6,gz1)],fill=(182,180,168))

# 墙
for i in range(len(pts)):
    u=pts[i];v=pts[(i+1)%len(pts)]
    ae=math.atan2(v[1]-u[1],v[0]-u[0]);lit=0.72+0.28*abs(math.cos(ae+ANG))
    q=[P(u[0],0,u[1]),P(v[0],0,v[1]),P(v[0],h,v[1]),P(u[0],h,u[1])]
    draw.polygon(q,fill=shade(COL["building"],lit))
# 屋顶
k=0.12;rise=min(2.6,max(1.8,h*0.15))
ip=[[cx+(q[0]-cx)*(1-k),cz+(q[1]-cz)*(1-k)] for q in pts]
for i in range(len(pts)):
    u=pts[i];v=pts[(i+1)%len(pts)];c2=ip[i];d2=ip[(i+1)%len(ip)]
    q=[P(u[0],h,u[1]),P(v[0],h,v[1]),P(d2[0],h+rise,d2[1]),P(c2[0],h+rise,c2[1])]
    draw.polygon(q,fill=shade(COL["roof"],0.82+0.2*((i%3)/3)))
qtop=[P(q[0],h+rise,q[1]) for q in ip];draw.polygon(qtop,fill=shade(COL["roof"],1.15))
draw.line(qtop+[qtop[0]],fill=(20,26,30,120),width=1)
draw.line([P(q[0],h,q[1]) for q in pts]+[P(pts[0][0],h,pts[0][1])],fill=(30,38,46,90),width=1)
# 马头墙：① 檐口压顶（每条边）
for i in range(len(pts)):
    A=pts[i];B=pts[(i+1)%len(pts)]
    dx=B[0]-A[0];dy=B[1]-A[1];L=math.hypot(dx,dy)
    if L<2:continue
    bot=[];top=[]
    for j in range(5):
        t=j/4;x=A[0]+dx*t;z=A[1]+dy*t;bot.append(P(x,h,z));top.append(P(x,h+1.0,z))
    draw.polygon(bot+top[::-1],fill=COL["maotou"],outline=(16,20,24))
# ② 近矩形：长轴两端山墙阶梯（马头墙）
n=len(pts);cx2=sum(q[0] for q in pts)/n;cz2=sum(q[1] for q in pts)/n
best,bA,bB=0.0,pts[0],pts[1]
for i in range(n):
    a=pts[i];b=pts[(i+1)%n];dd=math.hypot(b[0]-a[0],b[1]-a[1])
    if dd>best:best,bA,bB=dd,a,b
ux,uz=(bB[0]-bA[0])/best,(bB[1]-bA[1])/best;vx,vz=-uz,ux
us=[];vs=[]
for q in pts:
    us.append((q[0]-cx2)*ux+(q[1]-cz2)*uz); vs.append((q[0]-cx2)*vx+(q[1]-cz2)*vz)
umin,umax=min(us),max(us); vmin,vmax=min(vs),max(vs)
def pip(px,pz,poly):
    inside=False;N=len(poly);j=N-1
    for i2 in range(N):
        xi,zi=poly[i2];xj,zj=poly[j]
        if (zi>pz)!=(zj>pz) and px<(xj-xi)*(pz-zi)/(zj-zi+1e-12)+xi: inside=not inside
        j=i2
    return inside
# ② 长轴两端阶梯山墙：先找端面实际覆盖区段，再逐盒包含复查
M=5;base=1.2;step=1.6;thick=1.4;SAMP=80
for u_end,e in ((umax,1),(umin,-1)):
    ui=u_end-e*thick*0.5
    flags=[pip(cx2+ui*ux+(vmin+(vmax-vmin)*s/SAMP)*vx,cz2+ui*uz+(vmin+(vmax-vmin)*s/SAMP)*vz,pts) for s in range(SAMP+1)]
    bs=bl=cs=cl=0
    for s,f in enumerate(flags+[False]):
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
            if not pip(cx2+ui*ux+vv*vx,cz2+ui*uz+vv*vz,pts): ok=False;break
        if not ok: continue
        ccx=cx2+u_end*ux+vc*vx; ccz=cz2+u_end*uz+vc*vz; hu=thick*0.5; hv=vseg*0.49
        cs2=[[-hu,-hv],[hu,-hv],[hu,hv],[-hu,hv]]
        o=[P(ccx+s2[0]*ux+s2[1]*vx,h,ccz+s2[0]*uz+s2[1]*vz) for s2 in cs2]
        t=[P(ccx+s2[0]*ux+s2[1]*vx,h+lvl,ccz+s2[0]*uz+s2[1]*vz) for s2 in cs2]
        draw.polygon(o+[t[3],t[2],t[1],t[0]],fill=COL["maotou"],outline=(16,20,24))
for (A,B) in long_edges(pts):
    dx=B[0]-A[0];dy=B[1]-A[1];L=math.hypot(dx,dy)
    if L<2:continue
    bot=[];top=[]
    for i in range(5):
        t=i/4;x=A[0]+dx*t;z=A[1]+dy*t;bot.append(P(x,h,z));top.append(P(x,h+0.5,z))
    draw.polygon(bot+top[::-1],fill=COL["maotou"],outline=(16,20,24))
img.save(OUT)
print("saved",OUT,"building h=",h,"area=",round(a0))
