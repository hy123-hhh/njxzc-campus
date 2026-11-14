# 模型产物

本目录的模型由云端 GitHub Actions 自动生成（push 后触发 `.github/workflows/` 下的构建流程），并随构建自动发布到 GitHub Pages。

## 产物

| 文件 | 说明 | 状态 |
|---|---|---|
| **`campus_textured.glb`** | **当前正式模型（看这个）**：真实规划图地面 + 徽派楼栋分色（深灰/蓝灰/橙黄顶）+ 立体操场（跑道环+绿茵场）+ 16F 高楼 + 1454 棵树 + 八大功能页签场景，约 10.9MB | ✅ 最新 |
| `campus_geometry.glb` | 几何提取中间产物：纯色体块（无贴图地面），供建模管线校验轮廓用 | 中间产物 |
| `materials_report.json` / `geometry_report.json` | 构建报告：材质、三角面数、体积等 | 报告 |

> 注意：`campus_layout.glb`（布局级白模，2026-11-11 的中间版本）已移除，避免与正式模型混淆。查看模型请认准 **campus_textured.glb**。

## 生成方式

推送 `scripts/` 下的建模脚本（extract_outlines / refine_geometry / fix_layout / extract_trees / build_geometry / build_materials）会自动触发云端构建：重跑提取与修正管线 → 生成 GLB 与离线预览 → 自动提交产物 → 部署 GitHub Pages。

本地离线预览：`docs/离线预览.html`（与 GLB 同源生成，双击即可看，不需要联网）。

## 在线查看

GitHub Pages 地址（构建成功后自动更新）：
`https://hy123-hhh.github.io/njxzc-campus/docs/离线预览.html`

## 精度声明

坐标来自官方总平面图逐像素提取（100m 比例尺定标），属**布局级示意，非测绘坐标**。成果为教学展示用途，不作导航或工程依据。
