# 模型产物

本目录的模型**只由云端 GitHub Actions 生成**（`.github/workflows/build-model.yml`），本地不生成、不改动任何模型文件，也不在本地运行渲染或构建。

## 产物

| 文件 | 说明 |
|---|---|
| `campus_layout.glb` | glTF 2.0 二进制模型，布局级白模（主要建筑、道路、绿化、水体、运动场、地形底板） |
| `layout_report.json` | 生成报告：元素数、三角面数、坐标范围、文件体积 |

## 生成方式

推送 `data/layout.csv` 或 `scripts/build_layout.py` 会自动触发云端构建；也可在仓库 Actions 页面手动 **Run workflow**。

生成脚本 `scripts/build_layout.py` 只用 Python 标准库，无第三方依赖，不需要 Blender 或 GPU。

## 精度声明

坐标来自 `data/layout.csv`，按 R1 彩色总平面规划图目测归一化，属**布局级示意，非测绘坐标**。成果为教学展示用途，不作导航或工程依据。若与参考图有偏差，修改 CSV 后推送即可重新生成。
