# -*- coding: utf-8 -*-
"""生成「楼栋指认」页：把中文名与精细几何楼栋（building_id）人工绑定一次。

为什么需要这一步：
  - 舆情事件按「地点线索」定位到中文名（行知楼/南食堂/A区宿舍…），名字来自 layout.csv；
  - 精细几何模型（官方总平面图逐像素提取）用的是 building_025a…025l2 这类 id；
  - 官方总平面图上没有中文标注，而手绘图（R3）的方位与 R1 几何存在矛盾
    （见 docs/参考图核对表.md），因此"名字↔几何"无法靠坐标自动推定。

做法：本页为每个中文名列出坐标最近的 3 个候选楼栋（附官方总平面图裁剪缩略图），
      由人工点选确认一次，导出 name→building_id 映射 JSON；
      后续舆情即可精确落到精细几何模型上着色/定位。

用法：python scripts/gen_name_mapping_ui.py
输出：docs/楼栋指认.html（自包含单文件）+ data/name_mapping_candidates.json
"""
import base64
import csv
import io
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTLINE = BASE / "data" / "outlines.json"
LAYOUT = BASE / "data" / "layout.csv"
PLAN = BASE / "refs" / "official_01_plan_d.jpg"
OUT = BASE / "docs" / "楼栋指认.html"
CANDS = BASE / "data" / "name_mapping_candidates.json"

WORK_W = 1800.0
PX_PER_M = 1.81
PYC = 2393 / 2.0
TOP_N = 3
THRESH = 70.0     # 候选距离阈值（米）


def parent_id(fid: str) -> str:
    """025a3 → 025a（与 build_materials.building_identity 同规则）"""
    if not fid:
        return fid
    import re
    m = re.fullmatch(r"(building_025[a-dg-j])[1-4]", fid)
    return m.group(1) if m else fid


def main():
    import cv2
    ol = json.load(open(OUTLINE, encoding="utf-8"))
    blds = [p for p in ol["polygons"] if p["class"] == "building"]
    # 父楼栋聚合（同一楼栋的 4 段环带合并）
    parents = {}
    for p in blds:
        pid = parent_id(p.get("id") or "")
        g = parents.setdefault(pid, {"ids": [], "pts": [], "area": 0.0})
        g["ids"].append(p.get("id"))
        g["pts"].extend(p["points"])
        g["area"] += float(p.get("area_px") or 0)
    for pid, g in parents.items():
        g["cx"] = sum(q[0] for q in g["pts"]) / len(g["pts"])
        g["cz"] = sum(q[1] for q in g["pts"]) / len(g["pts"])
        g["size"] = max(12, int(math.sqrt(g["area"]) / 1.6))

    plan = cv2.imread(str(PLAN))
    H, W = plan.shape[:2]
    s = W / WORK_W

    def crop_b64(cx, cz, size_m):
        px = int((900 + cx * PX_PER_M) * s)
        py = int((PYC + cz * PX_PER_M) * s)
        half = int(max(40, size_m * PX_PER_M * s * 1.15))
        x0, x1 = max(0, px - half), min(W, px + half)
        y0, y1 = max(0, py - half), min(H, py + half)
        crop = plan[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        crop = cv2.resize(crop, (150, 150))
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return base64.b64encode(buf).decode() if ok else ""

    rows = [r for r in csv.DictReader(open(LAYOUT, encoding="utf-8-sig")) if r["kind"] == "building"]
    items = []
    for r in rows:
        lx, lz = float(r["x"]), float(r["z"])
        ranked = sorted(parents.items(),
                        key=lambda kv: math.hypot(kv[1]["cx"] - lx, kv[1]["cz"] - lz))
        cands = []
        for pid, g in ranked[:TOP_N]:
            d = math.hypot(g["cx"] - lx, g["cz"] - lz)
            cands.append({"id": pid, "dist": round(d, 1),
                          "thumb": crop_b64(g["cx"], g["cz"], g["size"]),
                          "cx": round(g["cx"], 1), "cz": round(g["cz"], 1),
                          "ok": d <= THRESH})
        items.append({"name": r["name"], "layout": [lx, lz], "floors": r["floors"],
                      "detail": r.get("detail", ""), "candidates": cands})

    payload = {"threshold_m": THRESH, "items": items}
    CANDS.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    js = json.dumps(items, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>楼栋指认 · 舆情空间绑定</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:"Microsoft YaHei",sans-serif;background:#f4f6f9;color:#1e293b}
header{background:#0f2a4a;color:#fff;padding:15px 22px}
header h1{margin:0;font-size:18px}
header p{margin:6px 0 0;font-size:12.5px;color:#a8c4de}
.wrap{display:grid;grid-template-columns:320px 1fr;gap:14px;padding:14px 18px 50px;max-width:1400px;margin:0 auto}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:13px 15px}
.list{max-height:640px;overflow:auto}
.row{padding:9px 11px;border:1px solid #e8edf3;border-radius:8px;margin-bottom:8px;cursor:pointer;background:#fcfdff}
.row:hover{border-color:#94a3b8}
.row.done{background:#f0fdf4;border-color:#86efac}
.row .n{font-size:14px;font-weight:700}
.row .m{font-size:11.5px;color:#64748b;margin-top:3px}
.pick{margin-bottom:20px;padding-bottom:14px;border-bottom:1px dashed #e2e8f0}
.pick h3{margin:0 0 4px;font-size:15px}
.pick .sub{font-size:12px;color:#64748b;margin-bottom:9px}
.cands{display:flex;gap:12px;flex-wrap:wrap}
.cand{border:1.5px solid #e2e8f0;border-radius:9px;padding:9px;width:170px;cursor:pointer;background:#fff}
.cand:hover{border-color:#2f6fed}
.cand.sel{border-color:#0f2a4a;box-shadow:0 0 0 2px #0f2a4a22}
.cand img{width:150px;height:150px;object-fit:cover;border-radius:6px;display:block;background:#eef2f7}
.cand .id{font-size:12px;font-weight:700;margin-top:6px}
.cand .d{font-size:11px;color:#94a3b8}
.bar{position:sticky;top:0;background:#0f2a4a;color:#fff;padding:9px 22px;display:flex;gap:14px;align-items:center;font-size:13px;z-index:9}
.bar button{background:#fbbf24;border:none;color:#0f2a4a;font-weight:700;padding:7px 14px;border-radius:7px;cursor:pointer}
.bar button.ghost{background:#ffffff22;color:#fff}
.note{font-size:12px;color:#64748b;line-height:1.8;margin-top:10px}
</style></head><body>
<header><h1>楼栋指认 · 舆情空间绑定</h1>
<p>为每个中文名选出它在精细几何模型里的对应楼栋（building_id）。官方图无中文标注、手绘图方位与几何矛盾，故需人工确认一次。</p></header>
<div class="bar"><span id="stat">加载中…</span>
<button onclick="exportJson()">导出映射 JSON</button>
<button class="ghost" onclick="copyJson()">复制到剪贴板</button>
<button class="ghost" onclick="localStorage.removeItem('nm_map');location.reload()">清空重来</button></div>
<div class="wrap">
 <div class="card"><div class="list" id="list"></div>
   <div class="note">候选按示意坐标就近排序，距离仅作参考；<b>以缩略图为准</b>。
   完成后把导出的 JSON 放到 <code>data/name_id_map.json</code>（可覆盖同名文件），舆情即可精确落到精细几何。</div></div>
 <div class="card" id="detail"></div>
</div>
<script>
const ITEMS = __PAYLOAD__;
let map = JSON.parse(localStorage.getItem('nm_map') || '{}');
const detail = document.getElementById('detail');
function renderList(){
  const done = Object.keys(map).length;
  document.getElementById('stat').textContent = `已确认 ${done} / ${ITEMS.length}`;
  document.getElementById('list').innerHTML = ITEMS.map((it,i)=>
    `<div class="row ${map[it.name]?'done':''}" data-i="${i}">
      <div class="n">${it.name}</div>
      <div class="m">布局坐标 (${it.layout[0]}, ${it.layout[1]}) ｜ ${it.floors}层 ${it.detail||''}
      ${map[it.name]?'｜ → '+map[it.name]:''}</div></div>`).join('');
  document.querySelectorAll('.row').forEach(r=>r.onclick=()=>show(+r.dataset.i));
}
function show(i){
  const it = ITEMS[i];
  detail.innerHTML = `<div class="pick"><h3>${it.name}</h3>
    <div class="sub">布局示意坐标 (${it.layout[0]}, ${it.layout[1]}) ｜ ${it.floors}层 ｜ ${it.detail||''}</div>
    <div class="cands">${it.candidates.map(c=>`
      <div class="cand ${map[it.name]===c.id?'sel':''}" data-id="${c.id}">
        <img src="data:image/jpeg;base64,${c.thumb}" alt="${c.id}">
        <div class="id">${c.id}</div><div class="d">示意距离 ${c.dist} m${c.ok?'':'（超阈值）'}</div>
      </div>`).join('')}</div></div>`;
  detail.querySelectorAll('.cand').forEach(c=>c.onclick=()=>{
    map[it.name]=c.dataset.id; localStorage.setItem('nm_map',JSON.stringify(map));
    renderList(); show(i);
  });
}
function exportJson(){
  const out = {};
  for (const it of ITEMS){ if(map[it.name]) out[it.name]={building_id:map[it.name], layout:it.layout, confirmed_by:'人工指认'}; }
  const blob = new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='name_id_map.json'; a.click();
}
function copyJson(){
  const out = {};
  for (const it of ITEMS){ if(map[it.name]) out[it.name]={building_id:map[it.name], layout:it.layout, confirmed_by:'人工指认'}; }
  navigator.clipboard.writeText(JSON.stringify(out,null,1)).then(()=>alert('已复制，可直接粘到 data/name_id_map.json'));
}
renderList(); if(ITEMS.length) show(0);
</script></body></html>
"""
    OUT.write_text(html.replace("__PAYLOAD__", js), encoding="utf-8", newline="\n")
    print(f"已生成 {OUT}（{len(items)} 个名称 × {TOP_N} 候选）")
    print(f"候选表 {CANDS}")


if __name__ == "__main__":
    main()
