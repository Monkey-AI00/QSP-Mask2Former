# Charge Port Demo

最小复现版充电口感知与探索 demo。

这个工程用于演示一条完整但最小化的闭环流程：

`RGB-D观测 -> 车型/关键点感知 -> 置信度评估 -> 低置信度触发探索 -> 语义图谱推理充电口 -> 输出动作与可视化`

当前版本严格按最小工程骨架搭建，使用 1 个车型 `car_A`、3 个视角和 3 个关键点，支持：

- 视角切换
- 2D / 伪 3D 充电口推理
- 帧图、日志、视频输出

## 目录结构

```text
charge_port_demo/
├─ data/
│  ├─ car_A/
│  │  ├─ meta.json
│  │  ├─ view_far_rgb.jpg
│  │  ├─ view_far_depth.png
│  │  ├─ view_far_kpts.json
│  │  ├─ view_mid_rgb.jpg
│  │  ├─ view_mid_depth.png
│  │  ├─ view_mid_kpts.json
│  │  ├─ view_close_rgb.jpg
│  │  ├─ view_close_depth.png
│  │  └─ view_close_kpts.json
│  └─ graphs/
│     └─ car_A_graph.json
├─ outputs/
│  ├─ frames/
│  ├─ logs/
│  └─ demo_video.mp4
├─ src/
│  ├─ config.py
│  ├─ load_data.py
│  ├─ perception.py
│  ├─ confidence.py
│  ├─ active_view.py
│  ├─ geometry.py
│  ├─ semantic_graph.py
│  ├─ infer_port.py
│  ├─ planner.py
│  ├─ visualize.py
│  └─ main.py
└─ requirements.txt
```

## 运行方式

先进入工程目录：

```bash
cd src/detectron2/charge_port_demo
```

安装依赖：

```bash
pip install -r requirements.txt
```

运行 demo：

```bash
python src/main.py --car_id car_A --init_view_id view_far
```

也可以从中距离或近距离启动：

```bash
python src/main.py --car_id car_A --init_view_id view_mid
python src/main.py --car_id car_A --init_view_id view_close
```

如果某个车型目录下暂时只有 `view_*_rgb.png` 或 `view_*_rgb.jpg`，程序会在首次运行时自动补生成：

- `meta.json`
- `data/graphs/<car_id>_graph.json`
- `view_*_kpts.json`
- `view_*_depth.png`

## 当前默认流程

以 `view_far` 为起点时，预期流程如下：

1. 读取 `car_A` 的远视角 RGB、Depth、关键点和图谱
2. 使用 `perception.py` 输出车型分类概率与关键点
3. 使用 `confidence.py` 计算最终观测置信度
4. 如果 `final_conf < 0.60`，则进入探索模式
5. 使用 `active_view.py` 选择更优视角，当前样例会切到 `view_close`
6. 在 `view_close` 上重新感知
7. 使用 `infer_port.py` 根据关键点和语义图谱推理充电口位置
8. 使用 `planner.py` 生成作业动作
9. 使用 `visualize.py` 输出帧图、日志和视频

## 输出结果

运行成功后会在 `outputs/` 下生成：

- `outputs/frames/001_view_far_explore.png`
  - 远视角感知结果，可看到当前模式为 `explore`
- `outputs/frames/002_view_close_work.png`
  - 近视角感知结果，可看到推理出的充电口位置和作业动作
- `outputs/logs/run_car_A_view_far.json`
  - 一次完整流程的结构化日志
- `outputs/demo_video.mp4`
  - 根据帧图合成的简单 demo 视频

## 模块说明

### `src/config.py`

集中管理配置项：

- 数据路径
- 输出路径
- 置信度阈值
- 分类和关键点的加权系数
- 默认相机内参
- 是否启用伪 3D 推理

### `src/load_data.py`

负责读取：

- 车型元数据 `meta.json`
- 单视角 RGB 图
- 深度图
- 关键点 JSON
- 图谱 JSON

这个模块只负责读数据，不包含推理逻辑。

### `src/perception.py`

负责模拟感知结果，当前最小版直接复用 `view_xxx_kpts.json` 中的标注/伪预测：

- `vehicle_id`
- `cls_probs`
- `keypoints_2d`
- `keypoint_scores`

后续如果要替换成真实网络输出，优先改这个模块。

### `src/confidence.py`

负责计算三类置信度：

- 分类置信度 `cls_conf`
- 关键点置信度 `kpt_conf`
- 最终观测置信度 `final_conf`

当前公式为：

```text
final_conf = alpha * cls_conf + beta * kpt_conf
```

### `src/active_view.py`

负责在低置信度时选择下一最佳观测点。

当前最小版使用简单打分策略：

- `close > mid > far`
- `front > side > oblique`
- 已知关键点越多越好
- 平均关键点分数越高越好

### `src/geometry.py`

负责几何换算：

- 从深度图采样深度
- 从像素坐标恢复相机坐标
- 预留 `camera_to_world()` 接口

当前实现的是伪 3D 最小版。

### `src/semantic_graph.py`

负责管理语义图谱关系，当前图谱里定义了：

- 可观测节点：`logo`、`left_headlight`、`right_headlight`
- 推理节点：`charge_port_center`
- 每个关键点到充电口的 2D / 3D 相对偏移

### `src/infer_port.py`

负责根据关键点和图谱推理充电口位置。

支持两种方式：

- `infer_port_2d()`
- `infer_port_3d()`

当前最小版逻辑是：

1. 每个关键点根据图谱关系得到一个充电口候选
2. 使用关键点分数做加权平均
3. 输出最终 `port_2d` 和可选的 `port_3d`

### `src/planner.py`

负责把推理结果转成动作：

- 探索模式：输出目标视角
- 作业模式：输出目标位置和姿态占位字段

### `src/visualize.py`

负责可视化相关工作：

- 画关键点和分数
- 画置信度面板
- 画充电口位置
- 画动作面板
- 保存帧图
- 从帧图合成视频

### `src/main.py`

主程序入口，只负责流程编排：

- 加载数据
- 运行感知
- 计算置信度
- 判断探索或作业
- 推理充电口
- 保存结果

## 数据文件说明

### `data/car_A/meta.json`

描述车型基础信息和可用视角。

### `data/car_A/view_*_kpts.json`

描述单个视角下的感知输入，包括：

- 关键点坐标
- 关键点分数
- 车型分类概率
- 视角质量标签

### `data/graphs/car_A_graph.json`

描述从观测关键点到充电口的关系，是推理模块的核心先验。

## 已验证结果

当前工程已实际运行通过，默认起点 `view_far` 的结果为：

- 第 1 步：`view_far`
  - `final_conf = 0.518`
  - 模式：`explore`
  - 下一视角：`view_close`
- 第 2 步：`view_close`
  - `final_conf = 0.898`
  - 模式：`work`
  - 推理出的 `port_2d ≈ [754.89, 363.36]`

## 后续扩展建议

- 把占位 RGB/Depth 图替换成真实采集图像
- 在 `perception.py` 中接入真实检测/关键点网络
- 在 `active_view.py` 中加入更真实的信息增益或历史视角收益
- 在 `semantic_graph.py` 中扩展更多车型和更多部件节点
- 在 `planner.py` 中接入真实机器人控制接口
