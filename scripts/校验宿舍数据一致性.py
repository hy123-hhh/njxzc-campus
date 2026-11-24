# -*- coding: utf-8 -*-
"""宿舍数据一致性校验：守住"单一真源 data/dorm_shapes.json"。

背景：2026-11-24 GPT 在两个文件里各存了一份 30 栋宿舍的 id 清单
      （index.html 的 const dorms、scripts/build_materials.py 的 const dormIds），
      两处手工维护 → 迟早分叉。本脚本以 data/dorm_shapes.json 为真源，
      比对两处派生数据，不一致即报错退出码 1。

用法：
  python scripts/校验宿舍数据一致性.py            # 校验（CI/验收可调）
  python scripts/校验宿舍数据一致性.py --sync    # 以 JSON 为准回写两处（谨慎使用）
"""
import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRUTH = BASE / "data" / "dorm_shapes.json"
INDEX = BASE / "index.html"
BUILD = BASE / "scripts" / "build_materials.py"


def read_json(path):
    return json.load(open(path, encoding="utf-8"))


def extract_index_dorms():
    s = INDEX.read_text(encoding="utf-8")
    m = re.search(r"const dorms = (\{.*?\});", s, re.S)
    return (json.loads(m.group(1)) if m else None), s


def extract_build_dormids():
    s = BUILD.read_text(encoding="utf-8")
    m = re.search(r"const dormIds=(\{.*?\});", s, re.S)
    return (json.loads(m.group(1)) if m else None), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="以 JSON 为准回写两处")
    a = ap.parse_args()

    truth = read_json(TRUTH)
    tb = truth["buildings"]
    tc = truth["center_to_id"]
    dorms, idx_src = extract_index_dorms()
    dormids, bld_src = extract_build_dormids()
    problems = []

    # 1) index.html 的 dorms
    if dorms is None:
        problems.append("index.html 中找不到 const dorms")
    else:
        ids_truth = set(tb)
        ids_page = set(dorms)
        if ids_truth != ids_page:
            problems.append(f"index.html dorms 与真源不一致："
                            f"多 {sorted(ids_page-ids_truth)[:6]} / 缺 {sorted(ids_truth-ids_page)[:6]}")
        for bid, v in tb.items():
            if bid in dorms:
                if dorms[bid][0] != v["floors"]:
                    problems.append(f"{bid} 层数不一致：页面 {dorms[bid][0]} vs 真源 {v['floors']}")
                if [[round(c[0], 2), round(c[1], 2)] for c in dorms[bid][1]] != \
                   [[round(c[0], 2), round(c[1], 2)] for c in v["outline"]]:
                    problems.append(f"{bid} 轮廓坐标不一致")
    # 2) build_materials.py 的 dormIds
    if dormids is None:
        problems.append("build_materials.py 中找不到 const dormIds")
    else:
        if set(dormids.values()) != set(tc.values()):
            problems.append("build_materials.py dormIds 的 id 集合与真源不一致")
        for k, v in tc.items():
            if k in dormids and dormids[k] != v:
                problems.append(f"质心 {k} 映射不一致：{dormids[k]} vs {v}")

    if a.sync and problems:
        print("[sync] 以 data/dorm_shapes.json 为准回写两处 …")
        new_dorms = {k: [v["floors"], v["outline"]] for k, v in sorted(tb.items())}
        idx_src = re.sub(r"const dorms = \{.*?\};",
                         "const dorms = " + json.dumps(new_dorms, ensure_ascii=False, separators=(",", ":")) + ";",
                         idx_src, count=1, flags=re.S)
        INDEX.write_text(idx_src, encoding="utf-8", newline="\n")
        bld_src = re.sub(r"const dormIds=\{.*?\};",
                         "const dormIds=" + json.dumps(tc, ensure_ascii=False, separators=(",", ":")) + ";",
                         bld_src, count=1, flags=re.S)
        BUILD.write_text(bld_src, encoding="utf-8", newline="\n")
        print("[sync] 已回写，重新校验 …")
        return main.__wrapped__() if hasattr(main, "__wrapped__") else _recheck()

    if problems:
        print("=== 宿舍数据不一致 ===")
        for p in problems[:20]:
            print("  [FAIL]", p)
        print(f"共 {len(problems)} 处问题。处理：改 data/dorm_shapes.json 后跑 --sync，或手工同步两处。")
        sys.exit(1)
    print(f"[ok] 宿舍数据一致：{len(tb)} 栋宿舍（层数 {sorted({v['floors'] for v in tb.values()})}）、"
          f"{len(tc)} 条质心映射，与 data/dorm_shapes.json 完全吻合")


def _recheck():
    dorms, _ = extract_index_dorms()
    dormids, _ = extract_build_dormids()
    truth = read_json(TRUTH)
    ok = set(dorms) == set(truth["buildings"]) and set(dormids.values()) == set(truth["center_to_id"].values())
    print("[ok] 回写后校验通过" if ok else "[FAIL] 回写后仍不一致")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
