# -*- coding: utf-8 -*-
"""生成「校园空间态势」页：把舆情事件落到校园平面楼栋上（3D 校园 × 舆情系统融合）。

输入：campus_repo/data/spatial_alerts.json（由 WSL 库导出：楼栋索引 + 各区域预警统计与明细）
输出：campus_repo/docs/校园空间态势.html（自包含单文件，可离线打开）

口径说明（页面内也会写明）：
  - 楼栋位置来自 3D 建模使用的官方总平面图提取结果；
  - 舆情位置为「文本线索推断的疑似发生地」，用于态势观察，不作为责任认定依据；
  - 预警含仿真演练数据与公开热榜数据，比例见页面图例。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "spatial_alerts.json"
OUT = BASE / "docs" / "校园空间态势.html"

d = json.load(open(DATA, encoding="utf-8"))
buildings = d["buildings"]
alerts = d["alerts_by_place"]

LEVEL_COLOR = {"red": "#d1384a", "orange": "#e07b39", "yellow": "#d9a441", "none": "#9aa7b4"}
LEVEL_NAME = {"red": "红色（危机）", "orange": "橙色（澄清）", "yellow": "黄色（观察）"}


def zone_center(name):
    """按名称在楼栋索引里找中心（同名多栋取平均）。"""
    hit = [b for b in buildings if b["name"] == name]
    if not hit:
        # 区域级名称 → 用关键字匹配（如"宿舍区""教学区"）
        key = name.replace("区", "").replace("生活区", "")
        hit = [b for b in buildings if key and key in b["name"]]
    if not hit:
        return None
    return (sum(b["x"] for b in hit) / len(hit), sum(b["z"] for b in hit) / len(hit))


REGIONS = d.get("regions", {})
zones = []
for name, g in alerts.items():
    members = REGIONS.get(name) or [name]
    hit = [b for b in buildings if b["name"] in members]
    if not hit:
        c = zone_center(name)
        if not c:
            continue
        hit = [{"name": name, "x": c[0], "z": c[1]}]
    lvl = "red" if g.get("red") else ("orange" if g.get("orange") else "yellow")
    zones.append({"name": name, "x": sum(b["x"] for b in hit) / len(hit),
                  "z": sum(b["z"] for b in hit) / len(hit),
                  "members": [b["name"] for b in hit], "level": lvl, **g})
zones.sort(key=lambda z: -z["n"])

payload = {"zones": zones, "buildings": buildings, "regions": REGIONS,
           "total_located": d["total_located"], "generated_at": d.get("generated_at")}
js = json.dumps(payload, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>校园空间态势 · 舆情claw</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#1e293b}
header{background:#0f2a4a;color:#fff;padding:16px 24px}
header h1{margin:0;font-size:19px;font-weight:800}
header p{margin:6px 0 0;font-size:12.5px;color:#a8c4de}
.wrap{display:grid;grid-template-columns:1fr 380px;gap:14px;padding:14px 18px 40px;max-width:1500px;margin:0 auto}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:9px 13px;min-width:118px}
.kpi .k{font-size:11.5px;color:#64748b}
.kpi .v{font-size:20px;font-weight:800;color:#0f2a4a;margin-top:2px}
#cv{width:100%;height:640px;background:#eef3f8;border-radius:8px;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#475569;margin-top:10px}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.items{max-height:560px;overflow:auto}
.item{border:1px solid #e8edf3;border-radius:8px;padding:9px 11px;margin-bottom:8px;background:#fcfdff}
.item .h{display:flex;gap:7px;align-items:center;font-size:12px;color:#64748b;margin-bottom:4px}
.badge{border-radius:3px;padding:1px 7px;font-size:11px;font-weight:700;color:#fff}
.badge.red{background:#d1384a}.badge.orange{background:#e07b39}.badge.yellow{background:#d9a441}
.badge.closed{background:#0f766e}.badge.open{background:#94a3b8}
.item .c{font-size:13px;line-height:1.65}
.hint{font-size:12px;color:#64748b;line-height:1.85;margin-top:10px}
h3.z{font-size:15px;margin:0 0 4px}
.zmeta{font-size:12px;color:#64748b;margin-bottom:10px}
</style>
</head>
<body>
<header>
  <h1>校园空间态势 · 舆情claw</h1>
  <p>舆情事件 × 校园空间关联视图（楼栋位置取自官方总平面图；舆情位置为文本线索推断的疑似发生地）</p>
</header>
<div class="wrap">
  <div>
    <div class="kpis" id="kpis"></div>
    <div class="card">
      <canvas id="cv"></canvas>
      <div class="legend">
        <span><i class="dot" style="background:#d1384a"></i>红色（危机级）</span>
        <span><i class="dot" style="background:#e07b39"></i>橙色（澄清/处置级）</span>
        <span><i class="dot" style="background:#d9a441"></i>黄色（观察级）</span>
        <span><i class="dot" style="background:#cbd5e1"></i>其他楼栋（本期无归集舆情）</span>
        <span style="color:#94a3b8">圆点大小＝该区域归集事件量</span>
      </div>
    </div>
    <div class="card" style="margin-top:12px;border-left:4px solid #d9a441">
      <div class="hint" id="caveat"></div>
      <div class="hint" style="color:#b45309;font-weight:700;margin-top:8px">
        ⚠ 位置口径：楼块位置取自布局白模轨道（<b>布局级示意坐标，非测绘成果</b>）；
        与精细几何模型（官方总平面图逐像素提取）的逐栋对应尚待人工指认（见 docs/楼栋指认.html）。
        本页用于展示系统的空间归属能力，不作为事件真实位置的认定依据。</div>
    </div>
  </div>
  <div class="card">
    <h3 class="z" id="zt">区域舆情明细</h3>
    <div class="zmeta" id="zm">点击左侧热力点查看</div>
    <div class="items" id="items"></div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const COL={red:"#d1384a",orange:"#e07b39",yellow:"#d9a441"};
const NAME={red:"红色（危机）",orange:"橙色（澄清）",yellow:"黄色（观察）"};
let sel=null;
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
function fit(){
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function bounds(){
  const xs=DATA.buildings.map(b=>b.x), zs=DATA.buildings.map(b=>b.z);
  return {x0:Math.min(...xs)-60,x1:Math.max(...xs)+60,z0:Math.min(...zs)-60,z1:Math.max(...zs)+60};
}
function toPx(b,x,z){
  const w=cv.clientWidth,h=cv.clientHeight;
  const sx=w/(b.x1-b.x0), sz=h/(b.z1-b.z0), s=Math.min(sx,sz);
  return [w/2+(x-(b.x0+b.x1)/2)*s, h/2+(z-(b.z0+b.z1)/2)*s];
}
function draw(){
  const b=bounds(); ctx.clearRect(0,0,cv.clientWidth,cv.clientHeight);
  // 道路/场地底纹（参考布局）
  ctx.fillStyle="#e6ecf3"; ctx.fillRect(0,0,cv.clientWidth,cv.clientHeight);
  // 楼栋（按所属区域着色：区域命中则用等级色，未命中用灰色）
  const zoneOf={};
  DATA.zones.forEach(z=>{ (z.members||[]).forEach(m=>{ zoneOf[m]=z; }); });
  DATA.buildings.forEach(bd=>{
    const [px,pz]=toPx(b,bd.x,bd.z);
    const z=zoneOf[bd.name];
    ctx.fillStyle=z?COL[z.level]:"#cbd5e1";
    ctx.strokeStyle="#94a3b8"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.rect(px-13,pz-9,26,18); ctx.fill(); ctx.stroke();
    ctx.fillStyle="#475569"; ctx.font="10px sans-serif"; ctx.textAlign="center";
    ctx.fillText(bd.name, px, pz+22);
  });
  // 热力点：画在该区域的每个成员楼上；数字徽标画在区域质心
  DATA.zones.forEach(z=>{
    const r=6+Math.min(14, Math.sqrt(z.n)*1.5);
    (z.members||[]).forEach(m=>{
      const bd=DATA.buildings.find(b=>b.name===m); if(!bd) return;
      const [px,pz]=toPx(b,bd.x,bd.z);
      ctx.beginPath(); ctx.arc(px,pz,r,0,Math.PI*2);
      ctx.fillStyle=COL[z.level]+"99"; ctx.fill();
      ctx.strokeStyle=COL[z.level]; ctx.lineWidth=1.5; ctx.stroke();
    });
    const [cx,cz]=toPx(b,z.x,z.z);
    ctx.beginPath(); ctx.arc(cx,cz,11,0,Math.PI*2);
    ctx.fillStyle=COL[z.level]; ctx.fill();
    ctx.strokeStyle="#fff"; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle="#fff"; ctx.font="bold 11px sans-serif"; ctx.textAlign="center";
    ctx.fillText(z.n, cx, cz+4);
    if(sel&&sel.name===z.name){ ctx.strokeStyle="#0f2a4a"; ctx.lineWidth=3; ctx.beginPath(); ctx.arc(cx,cz,15,0,Math.PI*2); ctx.stroke(); }
  });
}
cv.onclick=(ev)=>{
  const rect=cv.getBoundingClientRect(); const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;
  const b=bounds(); let best=null,bd=1e9;
  DATA.zones.forEach(z=>{
    (z.members||[]).forEach(m=>{
      const bb=DATA.buildings.find(x=>x.name===m); if(!bb) return;
      const [px,pz]=toPx(b,bb.x,bb.z); const d=Math.hypot(px-mx,pz-my); if(d<bd){bd=d;best=z;}
    });
    const [px,pz]=toPx(b,z.x,z.z); const d=Math.hypot(px-mx,pz-my); if(d<bd){bd=d;best=z;}
  });
  if(best&&bd<40){ sel=best; show(best); draw(); }
};
function show(z){
  document.getElementById("zt").textContent=z.name;
  document.getElementById("zm").innerHTML=
    `归集 <b>${z.n}</b> 条 ｜ 红 ${z.red} / 橙 ${z.orange} / 黄 ${z.yellow||0} ｜ 已闭环 <b>${z.closed}</b> 条 ｜ 最高等级 <span class="badge ${z.level}">${NAME[z.level]}</span>`;
  document.getElementById("items").innerHTML=(z.items||[]).map(it=>
    `<div class="item"><div class="h"><span class="badge ${it.level}">${NAME[it.level]||it.level}</span>
     <span class="badge ${it.status==='closed'?'closed':'open'}">${it.status==='closed'?'已闭环':'处置中'}</span>
     <span>#${it.id}</span><span>${(it.created_at||'').slice(5,16)}</span>
     ${it.handler?`<span>处置：${it.handler}</span>`:''}</div>
     <div class="c">${it.content}</div></div>`).join("")||'<div class="note">暂无明细</div>';
}
function kpis(){
  const tot=DATA.total_located||0;
  const red=DATA.zones.reduce((s,z)=>s+(z.red||0),0);
  const closed=DATA.zones.reduce((s,z)=>s+(z.closed||0),0);
  const covered=DATA.zones.length;
  const items=[["已定位事件",tot+" 条"],["覆盖区域",covered+" 处"],["红色事件",red+" 条"],
               ["区域闭环率",(closed/Math.max(tot,1)*100).toFixed(1)+"%"]];
  document.getElementById("kpis").innerHTML=items.map(x=>
    `<div class="kpi"><div class="k">${x[0]}</div><div class="v">${x[1]}</div></div>`).join("");
  document.getElementById("caveat").innerHTML=
    `<b>数据口径</b>：① 楼栋位置与名称取自 3D 建模所用的官方总平面图（30 个具名建筑/区域）；`
    +`② 舆情位置由正文地点线索推断（浴室/宿舍/食堂/快递点/操场等），为<b>疑似发生地</b>，不作为责任认定依据，`
    +`无可用线索的预警保持空值；③ 事件集合含仿真演练数据与公开热榜数据，真实按校名检索所得仅个位数，`
    +`本页展示的是系统<b>空间归属能力</b>而非真实舆情规模；④ 处置闭环为演练口径（handler 标注为「演练台账（系统补录）」）。`;
}
kpis(); fit(); window.addEventListener("resize",fit);
if(DATA.zones.length){ sel=DATA.zones[0]; show(sel); draw(); }
</script>
</body>
</html>
""".replace("__PAYLOAD__", js)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding="utf-8", newline="\n")
print(f"已生成 {OUT} （{len(html)//1024}KB，{len(zones)} 个热力区）")
