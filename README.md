# 南京晓庄学院（方山校区）三维建模资料库

公开参考图、布局数据、云端建模脚本与在线预览的统一仓库。

## 当前状态

| 阶段 | 状态 |
|---|---|
| 1. 公开参考图检索与整理 | 已完成（2026-11-10，用户已确认） |
| 2. GitHub 仓库 + 资料 + 在线预览 | 本仓库 |
| 3. 校区三维建模 | 由云端 Actions 生成，本地不执行 |

**本地约束**：本项目不在本地运行 Blender、渲染器或前端构建。本地只做轻量查看、资料整理与 git 提交。建模与 GLB 导出全部在云端（GitHub Actions）执行。

## 目录结构

```
refs/                 公开参考图原件（含高清底图与官方总体规划 PDF）
docs/                 参考图核对页、核对表、建模规范
data/layout.csv       布局元素清单与坐标（可编辑，用于校准）
scripts/build_layout.py  纯标准库 Python 建模脚本，生成 glTF 2.0 GLB
models/               云端生成的模型产物（本地不生成）
.github/workflows/    build-model.yml：云端建模 + Pages 部署
index.html            GitHub Pages 在线预览首页
```

## 参考图来源

| 编号 | 内容 | 来源 |
|---|---|---|
| R1 | 彩色总平面规划图（含比例尺、指北针、建筑编号图例） | 发展规划处 `fgc.njxzc.edu.cn/a1/1d/c8504a106781/page.htm` |
| R2 | 鸟瞰效果图 | 同上 |
| R3 | 手绘地图（建筑命名与方位） | 学生工作处 `xsc.njxzc.edu.cn/_t135/15/b0/c2336a71088/page.htm` |
| R4 | 方山校区总体规划 PDF（A0 单页） | 信息公开网 `xxgk.njxzc.edu.cn/d0/65/c6758a118885/page.htm` |
| R5 | 从 R4 无损抽出的 6622×9361 内嵌原图 | 由 R4 提取 |
| R6 | 百度地图 VR 全景（实景俯瞰，合规白名单内） | `scenes.map.baidu.com` |

完整来源、尺寸与用途见 `docs/参考图核对表.md` 与 `docs/参考图核对页.html`。

## 在线预览

仓库启用 GitHub Pages（Source = **GitHub Actions**）后，首页 `index.html` 提供：

- 左侧 R1 平面图底图、右侧 3D 模型并排对照
- 三张核心参考图画廊与高清原件下载
- 布局元素清单（读 `data/layout.csv`）
- 状态与合规说明

## 云端建模

推送 `data/layout.csv` 或 `scripts/build_layout.py` 会自动触发 `.github/workflows/build-model.yml`：

1. `python scripts/build_layout.py` 生成 `models/campus_layout.glb`（纯标准库，无重依赖）
2. 提交模型产物回仓库
3. 部署 GitHub Pages

也可在 Actions 页面手动 **Run workflow**。模型文件只由云端生成，本地 `models/` 不含任何模型。

## 如何校准布局

`data/layout.csv` 的坐标是**按 R1 目测归一化的布局级示意值，非测绘数据**。若与参考图有偏差，直接改 CSV 的 `x/z/w/d/h` 后推送，云端会重新生成模型。坐标系与字段说明见 `docs/建模规范.md`。

## 合规声明

- 仅使用学校官网及公开发布的宣传资料；未使用境外地图底图、卫星影像或测绘成果坐标。
- 后续若需地图底图，只使用腾讯地图 / 高德 / 百度 / 天地图，且不内嵌任何 API 密钥。
- 成果为**教学展示用途 · 示意性布局 · 非测绘成果**，不作导航或工程依据。
- 不标注涉密或敏感设施，不公开发布精确地理坐标。
- 图片版权归原发布方所有，本仓库仅作项目参考引用并逐张标注来源。

## 本地推送命令（首次）

```bash
cd campus_repo
git init
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱"
git add .
git commit -m "init: 方山校区参考图与云端建模管线"
git remote add origin https://github.com/<用户名>/<仓库名>.git
git push -u origin main
```

推送后在仓库 **Settings → Pages** 把 Source 设为 **GitHub Actions**，再手动运行一次 workflow 即可生成模型与在线预览。
