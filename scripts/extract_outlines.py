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
MIN_AREA = {"building": 520, "water": 600, "green": 1600, "road": 1800}


def classify_masks(bgr):
    b, g, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16), bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    warm = (r > g + 14) & (g > b + 4) & (r > 135) & (sat > 28)
    gray = (sat < 34) & (mx > 95) & (mx < 214)
    building = (warm | gray).astype(np.uint8) * 255
    water = ((b > r + 20) & (b > g + 6) & (b > 105) & (sat > 30)).astype(np.uint8) * 255
    green = ((g > r + 9) & (g > b + 7) & (g > 95) & (sat > 24)).astype(np.uint8) * 255
    road = ((mn > 222) & (sat < 26)).astype(np.uint8) * 255
    return {"building": building, "water": water, "green": green, "road": road}


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
