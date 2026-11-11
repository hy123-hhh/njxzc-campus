# -*- coding: utf-8 -*-
"""第三阶段：逐栋拆分 + 图上标尺定标（仅云端 GitHub Actions 运行）。

1) 检测官方总平面图左下角比例尺黑线长度，反推 px/m，重算世界坐标；
2) 用形态学腐蚀断开连体建筑，按楼间窄缝把成片色块拆成逐栋轮廓；
3) 覆盖 data/outlines.json 与 docs/outline_overlay.jpg，供后续几何/材质阶段复用。

输出：data/scale_info.json、data/refine_report.json、docs/scale_crop.png、
      data/outlines.json、docs/outline_overlay.jpg
"""
import json
import os
import cv2
import numpy as np

from extract_outlines import classify_masks, clean, MIN_AREA, WORK_W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "refs", "official_01_plan_d.jpg")
OUTLINES = os.path.join(ROOT, "data", "outlines.json")
SCALE_INFO = os.path.join(ROOT, "data", "scale_info.json")
REFINE_REPORT = os.path.join(ROOT, "data", "refine_report.json")
OVERLAY = os.path.join(ROOT, "docs", "outline_overlay.jpg")
SCALE_CROP = os.path.join(ROOT, "docs", "scale_crop.png")

SCALE_METERS = float(os.environ.get("SCALE_METERS", "100"))
ERODE_ITER = int(os.environ.get("ERODE_ITER", "2"))


def detect_scale(small):
    """在左下角找比例尺水平黑线，返回线段长度与裁剪放大图。"""
    h, w = small.shape[:2]
    x0, x1 = 0, int(0.34 * w)
    y0, y1 = int(0.78 * h), h
    roi = small[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark = (gray < 105).astype(np.uint8)
    rh, rw = dark.shape
    best = (0, 0, 0, 0)
    for yy in range(int(0.68 * rh), int(0.92 * rh)):
        xs = np.flatnonzero(dark[yy])
        if xs.size < 2:
            continue
        s, e = int(xs.min()), int(xs.max())
        span = e - s
        if 60 < span < 0.55 * rw and span > best[0]:
            best = (span, s, e, yy)
    crop = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return best[0], crop, (x0, y0, x1, y1)


def split_buildings(mask, min_area=420):
    """腐蚀断开连体建筑 -> 连通域 -> 膨胀恢复尺寸 -> 逐栋轮廓。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(mask, k, iterations=ERODE_ITER)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 90:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        grown = cv2.dilate(comp, k, iterations=ERODE_ITER)
        grown = cv2.bitwise_and(grown, mask)
        cs, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if a / max(bw * bh, 1) < 0.24:
                continue
            if bw / max(bh, 1) > 14 or bh / max(bw, 1) > 14:
                continue
            out.append(c)
    return out


def main():
    img = cv2.imread(PLAN, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("无法读取高清总平面图")
    H, W = img.shape[:2]
    h = int(H * WORK_W / W)
    small = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_AREA)

    line_px, crop, roi_box = detect_scale(small)
    cv2.imwrite(SCALE_CROP, crop)
    if line_px <= 0:
        raise SystemExit("未检测到比例尺线段，停止定标（避免产出错误比例）")
    px_per_m = line_px / SCALE_METERS
    scale = {"line_px": line_px, "assumed_meters": SCALE_METERS,
             "px_per_m": round(px_per_m, 4),
             "campus_width_m": round(WORK_W / px_per_m, 1),
             "campus_height_m": round(h / px_per_m, 1),
             "roi_px": list(roi_box)}
    json.dump(scale, open(SCALE_INFO, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def to_world(px, py):
        return [round((px - WORK_W / 2) / px_per_m, 2),
                round((py - h / 2) / px_per_m, 2)]

    masks = classify_masks(small)
    bmask = clean(masks["building"], 3)
    parts = split_buildings(bmask)
    overlay = small.copy()
    polys = []
    for i, c in enumerate(parts):
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.007 * peri, True)
        if len(ap) < 3:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bx > 0.70 * WORK_W and by > 0.66 * h:
            continue
        if bx < 0.24 * WORK_W and by > 0.84 * h:
            continue
        pts_px = [(float(p[0][0]), float(p[0][1])) for p in ap]
        pts = [to_world(px, py) for px, py in pts_px]
        polys.append({"id": f"building_{i:03d}", "class": "building",
                      "area_px": round(float(cv2.contourArea(c)), 1),
                      "bbox_px": [bx, by, bw, bh],
                      "centroid": to_world(bx + bw / 2, by + bh / 2),
                      "points": pts})
        cv2.drawContours(overlay, [ap], -1, (30, 90, 235), 2)
        cv2.putText(overlay, str(i), (int(bx + bw / 2), int(by + bh / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 60, 210), 1, cv2.LINE_AA)

    for cls, color in (("water", (220, 140, 30)), ("green", (60, 190, 60)), ("road", (190, 190, 40))):
        m = clean(masks[cls], 5)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            area = cv2.contourArea(c)
            if area < MIN_AREA[cls]:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx > 0.70 * WORK_W and by > 0.66 * h:
                continue
            if bx < 0.24 * WORK_W and by > 0.84 * h:
                continue
            if area / max(bw * bh, 1) < 0.30:
                continue
            ap = cv2.approxPolyDP(c, 0.007 * cv2.arcLength(c, True), True)
            if len(ap) < 3:
                continue
            pts = [to_world(float(p[0][0]), float(p[0][1])) for p in ap]
            polys.append({"id": f"{cls}_{len(polys):03d}", "class": cls,
                          "area_px": round(float(area), 1),
                          "bbox_px": [bx, by, bw, bh],
                          "centroid": to_world(bx + bw / 2, by + bh / 2),
                          "points": pts})
            cv2.drawContours(overlay, [ap], -1, color, 2)

    json.dump({"source": "refs/official_01_plan_d.jpg",
               "image": [W, H], "work": [WORK_W, h],
               "scale": scale,
               "note": "轮廓逐像素取自官方总平面图；建筑按楼间窄缝逐栋拆分；坐标由图上比例尺定标",
               "polygons": polys}, open(OUTLINES, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    cv2.imwrite(OVERLAY, overlay, [cv2.IMWRITE_JPEG_QUALITY, 86])
    rep = {"buildings_split": sum(1 for p in polys if p["class"] == "building"),
           "polygons_total": len(polys), "scale": scale, "erode_iter": ERODE_ITER}
    json.dump(rep, open(REFINE_REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("定标:", scale)
    print("拆分后建筑", rep["buildings_split"], "栋 | 总轮廓", len(polys))


if __name__ == "__main__":
    main()
