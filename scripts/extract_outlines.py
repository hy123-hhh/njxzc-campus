# -*- coding: utf-8 -*-
"""从官方高清总平面图提取真实轮廓（仅云端 GitHub Actions 运行）。

输入 refs/official_01_plan_d.jpg (4604x6123)
输出 data/outlines.json / data/extract_stats.json / docs/outline_overlay.jpg

原理：按颜色分离「建筑 / 水体 / 绿地 / 道路」，形态学去噪后做轮廓跟踪与
多边形简化，得到与官方总平面图 1:1 的真实形状，再线性映射到校园坐标系。
"""
import json
import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
OUT_JSON = os.path.join(ROOT, "data", "outlines.json")
OUT_STATS = os.path.join(ROOT, "data", "extract_stats.json")
OUT_OVERLAY = os.path.join(ROOT, "docs", "outline_overlay.jpg")

WORK_W = 1800
X_MIN, X_MAX = -350.0, 350.0
Z_MIN, Z_MAX = -450.0, 450.0
# 图例实测色（RGB）：新建建筑 salmon=(237,168,128) -> BGR 排序
SALMON_BGR = (117, 142, 227)
MIN_AREA = {"building": 300, "water": 600, "green": 1600, "road": 1800}


def classify_masks(bgr):
    """按规划图真实调色板分层（2026-11-11 依据图例色卡重校）：
    楼 = salmon 新建建筑（白底保留建筑由 building_union_mask 的封闭口袋负责）
    水 = 浅蓝 (138,164,212) 系；绿地 = 各档绿（黄扇形铺装 g<r 不会误入）；
    路 = 中灰/浅灰铺装 + 米黄色路面。"""
    b, g, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16), bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    salmon = (np.abs(b - SALMON_BGR[0]) + np.abs(g - SALMON_BGR[1]) + np.abs(r - SALMON_BGR[2])) < 100
    building = salmon.astype(np.uint8) * 255
    water = ((b > r + 40) & (b > g + 25) & (b > 185)).astype(np.uint8) * 255
    green = ((g > r + 12) & (g > b + 12)).astype(np.uint8) * 255
    road = (((sat < 30) & (mn > 118) & (mn < 216)) |
            ((r > g + 8) & (g > b + 8) & (r > 200) & (sat > 28) & (sat < 95))).astype(np.uint8) * 255
    return {"building": building, "water": water, "green": green, "road": road}


def building_union_mask(bgr):
    """最终建筑掩码 = 封闭白色口袋(保留建筑) ∪ salmon(新建建筑)。

    保留建筑是白底+深灰封闭描边：对白色掩码做形态学闭运算弥合文字/编号，
    再从图外 flood-fill，灌不到的白口袋即楼体；外圈补白边保证种子点在纸面。
    返回 (union_mask, pockets_mask) 便于诊断。"""
    h, w = bgr.shape[:2]
    b, g, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16), bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    white = ((mn > 205) & (sat < 26)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k, iterations=2)
    padded = np.pad(white, 2, constant_values=255)
    ff = padded.copy()
    seed = np.zeros((h + 6, w + 6), np.uint8)
    cv2.floodFill(ff, seed, (0, 0), 0)
    pockets = ff[2:-2, 2:-2]
    # 深灰屋顶斜线环（保留建筑的坡顶画法）：先开运算去掉 1-2px 的铺装格线，
    # 只留 >=3px 的斜线带；再只取白口袋周边 15px 内的深灰，绿色主导（树）排除。
    green_dom = (g > r + 15) & (g > b + 15)
    dark = ((mn < 115) & (mx < 130) & (sat < 70) & (~green_dom)).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k3, iterations=1)
    near = cv2.dilate(pockets, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    dark_near = cv2.bitwise_and(dark, near)
    salmon = (np.abs(b - SALMON_BGR[0]) + np.abs(g - SALMON_BGR[1]) + np.abs(r - SALMON_BGR[2])) < 100
    salmon = (salmon.astype(np.uint8)) * 255
    salmon = cv2.morphologyEx(salmon, cv2.MORPH_CLOSE, k3, iterations=1)
    union = cv2.bitwise_or(pockets, salmon)
    union = cv2.bitwise_or(union, dark_near)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k3, iterations=1)
    return union, pockets


def clean(mask, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m


def main():
    img = cv2.imread(PLAN, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("无法读取高清总平面图: " + PLAN)
    H, W = img.shape[:2]
    h = int(H * WORK_W / W)
    small = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_AREA)
    masks = classify_masks(small)
    overlay = small.copy()
    colors = {"building": (30, 90, 235), "water": (220, 140, 30),
              "green": (60, 190, 60), "road": (190, 190, 40)}
    polygons, stats = [], {"source": "refs/official_01_plan_d.jpg",
                           "image": [W, H], "work": [WORK_W, h],
                           "classes": {}}

    def to_world(px, py):
        return [round(X_MIN + px / WORK_W * (X_MAX - X_MIN), 2),
                round(Z_MIN + py / h * (Z_MAX - Z_MIN), 2)]

    for cls, mask in masks.items():
        m = clean(mask, 3 if cls == "building" else 5)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA[cls]:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx > 0.70 * WORK_W and by > 0.66 * h:
                continue
            if bx < 0.24 * WORK_W and by > 0.84 * h:
                continue
            bbox_area = max(bw * bh, 1)
            if area / bbox_area < 0.30:
                continue
            if bw / max(bh, 1) > 12 or bh / max(bw, 1) > 12:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.006 * peri, True)
            if len(approx) < 3:
                continue
            pts_px = [(float(p[0][0]), float(p[0][1])) for p in approx]
            pts = [to_world(px, py) for px, py in pts_px]
            bx, by, bw, bh = cv2.boundingRect(c)
            kept.append({
                "id": f"{cls}_{len(kept):03d}",
                "class": cls,
                "area_px": round(float(area), 1),
                "bbox_px": [bx, by, bw, bh],
                "centroid": to_world(bx + bw / 2, by + bh / 2),
                "points": pts,
            })
            cv2.drawContours(overlay, [approx], -1, colors[cls], 2)
            cx, cy = int(bx + bw / 2), int(by + bh / 2)
            cv2.putText(overlay, str(len(kept) - 1), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1, cv2.LINE_AA)
        polygons.extend(kept)
        stats["classes"][cls] = {
            "raw_pixels": int(np.count_nonzero(mask)),
            "contours_found": len(contours),
            "kept": len(kept),
            "total_area_px": round(sum(p["area_px"] for p in kept), 1),
        }

    stats["polygons_total"] = len(polygons)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_OVERLAY), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"source": "refs/official_01_plan_d.jpg",
                   "image": [W, H], "work": [WORK_W, h],
                   "coord_range": [X_MIN, X_MAX, Z_MIN, Z_MAX],
                   "note": "轮廓按官方高清总平面图逐像素提取，形状与参考图 1:1；坐标为线性映射的示意值，非测绘成果",
                   "polygons": polygons}, f, ensure_ascii=False, indent=1)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    cv2.imwrite(OUT_OVERLAY, overlay, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print("轮廓提取完成:", len(polygons), "个")
    for k, v in stats["classes"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
