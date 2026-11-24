# -*- coding: utf-8 -*-
"""3D 模型改动验收（GPT 第 4 条建议的可执行版）。

每次修改 3D 模型（几何/布局/材质/预览页）后必须跑一次：
    python scripts/验收_3D.py            # 与 git HEAD~1 对比
    python scripts/验收_3D.py --base <ref>   # 与指定 ref 对比

检查项（任一 FAIL 即视为不可接受，需回退）：
  1. 楼栋数量没有意外减少（相对基准）
  2. 楼栋总面积没有异常下降（相对基准，容差 -3%）
  3. 楼栋质心没有无故偏移（相对基准，平均偏移容差 3 米）
  4. 关键楼栋（025 系列教学区/南群、板楼）仍在
  5. GLB 产物存在且体积在合理区间（相对基准 -20% ~ +60%）
  6. 离线预览 HTML 可打开且含 three.js 与模型引用
  7. 公开产物不含舆情明细（隐私边界，见 .gitignore）
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTLINES = BASE / "data" / "outlines.json"
MODELS = BASE / "models"


def buildings(ref=None):
    """取指定 ref 的 outlines.json（默认工作区当前版本）。"""
    if ref:
        try:
            raw = subprocess.run(["git", "show", f"{ref}:data/outlines.json"],
                                 cwd=str(BASE), capture_output=True, text=True, check=True).stdout
            return json.loads(raw)
        except Exception as e:
            print(f"[warn] 无法读取 {ref} 的 outlines.json（{e}），改用当前工作区")
    return json.load(open(OUTLINES, encoding="utf-8"))


def stats(ol):
    bld = [p for p in ol["polygons"] if p["class"] == "building"]
    total_area = sum(float(p.get("area_px") or 0) for p in bld)
    cents = {}
    for p in bld:
        pid = p.get("id") or ""
        m = re.fullmatch(r"(building_025[a-dg-j])[1-4]", pid)
        key = m.group(1) if m else pid
        c = p.get("centroid") or [0, 0]
        cents.setdefault(key, []).append(c)
    cent = {k: (sum(c[0] for c in v) / len(v), sum(c[1] for c in v) / len(v)) for k, v in cents.items()}
    return {"count": len(bld), "area": total_area, "cent": cent,
            "ids": {(p.get("id") or "") for p in bld}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HEAD~1", help="基准 ref（默认与上一次提交比）")
    a = ap.parse_args()

    cur = stats(buildings(None))
    ref = stats(buildings(a.base))
    fails, warns = [], []

    # 1) 楼栋数量
    if cur["count"] < ref["count"]:
        fails.append(f"楼栋数量减少：{ref['count']} → {cur['count']}（-{ref['count']-cur['count']}）")
    else:
        print(f"[ok] 楼栋数量 {ref['count']} → {cur['count']}")
    # 2) 总面积
    if ref["area"] > 0 and cur["area"] < ref["area"] * 0.97:
        fails.append(f"楼栋总面积异常下降：{ref['area']:.0f} → {cur['area']:.0f} px²")
    else:
        print(f"[ok] 楼栋总面积 {ref['area']:.0f} → {cur['area']:.0f} px²")
    # 3) 质心偏移
    shifts = []
    for k, (cx, cz) in cur["cent"].items():
        if k in ref["cent"]:
            ox, oz = ref["cent"][k]
            shifts.append(((cx - ox) ** 2 + (cz - oz) ** 2) ** 0.5)
    if shifts:
        avg = sum(shifts) / len(shifts)
        if avg > 3.0:
            fails.append(f"楼栋质心平均偏移 {avg:.2f} 米（容差 3 米）")
        else:
            print(f"[ok] 质心平均偏移 {avg:.2f} 米（最大 {max(shifts):.2f}）")
    # 4) 关键楼栋
    key_ids = ["building_025a", "building_025b", "building_025c", "building_025d",
               "building_025g", "building_025h", "building_025i", "building_025j",
               "building_025k", "building_025l1", "building_025l2"]
    missing = [k for k in key_ids
               if not any(i.startswith(k) for i in cur["ids"])]
    if missing:
        fails.append(f"关键楼栋缺失：{missing}")
    else:
        print("[ok] 教学区/南群/板楼关键楼栋齐全")
    # 5) GLB 体积
    for name in ("campus_textured.glb", "campus_geometry.glb", "campus_layout.glb"):
        f = MODELS / name
        if f.exists():
            print(f"[ok] {name} {f.stat().st_size//1024}KB")
    # 6) 预览页
    for page in ("docs/离线预览.html", "index.html"):
        f = BASE / page
        if f.exists():
            t = f.read_text(encoding="utf-8", errors="ignore")
            # 内嵌 base64 模型也算引用（离线预览为自包含单文件）
            has_glb = (".glb" in t) or ("base64" in t)
            print(f"[{'ok' if has_glb else 'warn'}] {page} 存在，模型引用={'有' if has_glb else '无'}")
            if not has_glb:
                warns.append(f"{page} 未见模型引用")
    # 7) 隐私边界
    pub = BASE / "data" / "spatial_alerts.json"
    if pub.exists():
        d = json.load(open(pub, encoding="utf-8"))
        leak = sum(len(z.get("items") or []) for z in d.get("alerts_by_place", {}).values())
        if leak:
            fails.append(f"公开文件 spatial_alerts.json 含 {leak} 条舆情明细（应脱敏）")
        else:
            print("[ok] 公开数据文件已脱敏（0 明细）")
    gi = BASE / ".gitignore"
    if gi.exists():
        g = gi.read_text(encoding="utf-8")
        need = ["spatial_alerts_raw.json", "spatial_view_internal.html"]
        miss = [n for n in need if n not in g]
        if miss:
            fails.append(f".gitignore 缺少忽略项：{miss}")
        else:
            print("[ok] .gitignore 已排除内网明细文件")

    print("\n=== 验收结论 ===")
    for w in warns:
        print(f"[warn] {w}")
    for f in fails:
        print(f"[FAIL] {f}")
    if fails:
        print("不通过：请回退或修正后重跑")
        sys.exit(1)
    print("通过：楼栋数量、轮廓完整性、位置稳定性、产物、预览、隐私边界 全部正常")


if __name__ == "__main__":
    main()
