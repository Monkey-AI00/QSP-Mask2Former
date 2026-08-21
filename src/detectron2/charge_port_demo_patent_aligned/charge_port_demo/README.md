# 车辆充电口定位可复现仿真原型

本工程用于复现“多模态观测—置信度评估—主动视点规划—语义图谱刚体变换定位—动作输出”的闭环计算过程。它是专利提交前的研发佐证原型，不是训练完毕的生产模型，也不是实车或实体机器人精度报告。

## 能力范围

- RGB、深度图与由深度反投影得到的点云特征融合；
- 车辆分类概率的归一化信息熵置信度；
- 关键点高斯热力图及峰值置信度；
- 分类、热力图和不确定性的复合损失与确定性校准示例；
- 基于覆盖度、观察角度、遮挡先验和预测置信度增益的主动视点选择；
- 相机坐标到机器人基座坐标的 4×4 齐次变换；
- 车辆部件到充电口的语义图谱刚体变换融合；
- 推理数据和评测真值隔离；
- 三种车型的独立日志、帧图、动画、汇总和 SHA-256 清单。

## 数据边界

`data/*/view_*_kpts.json` 中的 `replay_output` 是固定回放头输出，用于确定性重放；`ground_truth` 只允许由 `evaluation.py` 在推理结束后读取。程序不会在正式运行时自动生成缺失输入。若要演示派生数据准备，可显式调用 `load_data.prepare_synthetic_assets()`，但派生数据不能冒充实测数据。

当前 RGB、深度、点云及关键点回放仅用于计算链路复现。`model3` 的近景样例包含二维人工仿真参考；其他案例未提供独立真值时，评测误差为 `null`，不会补造精度结论。

## 环境与运行

建议 Python 3.10 及以上版本。

```powershell
python -m pip install -r requirements.txt
python src/main.py --car_id car_A --init_view_id view_far
python src/run_validation.py
python -m unittest discover -s tests -v
```

单案例输出位于 `outputs/run_<car_id>_<view>/`；三案例验证输出位于：

```text
outputs/
├─ cases/car_A/
├─ cases/car_B/
├─ cases/model3/
├─ validation_summary.json
└─ manifest_sha256.json
```

若 OpenCV 可用，动画导出为 MP4；否则使用 Pillow 导出 GIF。两种后端都不影响定位数值。

## 主要模块

- `src/multimodal.py`：RGB/深度/点云确定性特征与注意力融合、热力图。
- `src/confidence.py`：信息熵、热力图峰值和加权置信度。
- `src/losses.py`：复合损失与固定种子校准。
- `src/active_view.py`：候选位姿可观测性与置信度增益排序。
- `src/geometry.py`、`src/validation.py`：投影、反投影及刚体变换校验。
- `src/semantic_graph.py`、`src/infer_port.py`：4×4 图谱关系与候选融合。
- `src/evaluation.py`：独立真值评测，禁止进入推理路径。
- `src/main.py`：单案例编排和结构化证据日志。
- `src/run_validation.py`：三案例汇总、环境记录和哈希清单。

## 可复现性

配置固定种子为 `20260727`。相同代码、输入数据和依赖版本下，结构化预测结果保持一致。帧图或视频编码字节可能随图像/视频库版本变化，因此以 JSON 日志、汇总和清单为主要复现证据。
