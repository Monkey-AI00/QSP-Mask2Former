# RealSense + AprilTag + YOLO：标定与坐标输出（Tag 作为世界坐标系）

## 1. 项目需求拆解（你现在要实现的“标定”是什么）

你描述的思路本质是在做一条**感知→坐标变换**链路，并用 AprilTag 模拟/替代“机器人基座坐标系”：

- **世界坐标系 World**：用固定在工装/桌面上的 AprilTag 坐标系代替（World Origin = Tag）。
- **相机坐标系 Cam**：RealSense 相机坐标系。
- **像素坐标系 Pixel**：彩色图像像素 (u,v)。

你要的最终输出通常是：**目标点在世界坐标系下的位置** \(X_world = (x,y,z)\)（单位米），用于后续机械臂/控制。

其中关键步骤：

1. YOLO 在彩色图上检测目标，得到框中心像素坐标 (u,v)。
2. RealSense 深度图对齐到彩色图后，在 (u,v) 处读取深度 z（米）。
3. 用相机内参把 (u,v,z) 反投影成相机坐标系 3D 点 \(X_cam\)。
4. AprilTag 检测 + 位姿估计得到 tag→cam 的外参（R,t）。
5. 把 \(X_cam\) 变换到 tag/world 坐标系 \(X_world (= X_tag)\)。

> 这一步（4-5）就是你现在说的“标定”：不是传统意义的“相机内参标定”（RealSense 已提供），而是**相机相对于世界（Tag）的外参估计**。

---

## 2. 你需要准备的参数

- **Tag 边长**（单位米）：例如 5cm 的 tag：`tag_size=0.05`
- **Tag 字典**：常用 `apriltag_36h11`（OpenCV 中对应 `DICT_APRILTAG_36h11`）
- （可选）只用某一个 tag：`tag_id=0/1/...`，避免画面中多个 tag 时跳变

---

## 3. 代码如何整合（推荐结构）

本仓库已新增/抽象出 3 个可复用模块：

- `ultralytics/yolo/yolo_utils.py`
  - `infer_detections()`：对一帧 BGR 图推理并返回检测框（含中心点 uv）
- `ultralytics/yolo/realsense_utils.py`
  - `start_pipeline()`：开启 RealSense
  - `frames_to_aligned_color_depth()`：对齐 depth→color，输出 (color_bgr, depth_frame)
  - `depth_at_uv()`：取深度 z（米）
  - `deproject_uv_depth_to_cam_xyz()`：反投影得到 \(X_cam\)
- `ultralytics/yolo/apriltag_utils.py`
  - `detect_apriltag_poses()`：检测 AprilTag 并 solvePnP 得到 (rvec,tvec)
  - `TagPose.cam_to_tag()`：把 \(X_cam\) 变到 \(X_tag\)

整合脚本：

- `ultralytics/yolo/rs_yolo_apriltag_world.py`
  - 实时输出 `tag_xyz`（米），可选 `--show` 可视化

---

## 4. 运行方式（实时：RealSense + AprilTag + YOLO）

在项目根目录执行：

```bash
cd /home/users1/sjw/cursor/Yolo_pointrend

python /home/user/sjw/Yolo_pointrend/ultralytics/yolo/rs_yolo_apriltag_world.py \
  --weights /home/user/sjw/Yolo_pointrend/ultralytics/runs/detect/yolo11_cable_box/weights/last.pt \
  --tag_size 0.05 \
  --tag_dict apriltag_25h9 \
  --tag_id 0 \
  --conf 0.25 \
  --device 0 \
  --show
  --print_every 10
```

输出示例（每帧最多 `--max_det` 个检测）：

```
[0] cls=0 conf=0.901 uv=(512.3,384.0)  cam_xyz=(0.012,-0.034,0.421)m  tag_xyz=(0.105,0.022,0.006)m  tag_id=0
```

---

## 5. 关键注意事项（避免踩坑）

- **必须 depth 对齐到 color**：否则 YOLO 的 (u,v) 与 depth 像素不对应，会导致 Z/3D 点错误。
- **Tag 尺寸必须准确**：pose 的尺度完全由 `tag_size` 决定，填错会整体缩放。
- **一个稳定的世界系**：Tag 必须固定不动；画面里最好固定只用一个 tag id。
- **深度空洞/盲区**：`get_distance()` 返回 0 代表无效，脚本会跳过该点。
- **后续机械臂集成**：未来只要再乘一个固定变换 \(T_{tag\to base}\) 就能把 `tag_xyz` 变到机械臂基座系。


