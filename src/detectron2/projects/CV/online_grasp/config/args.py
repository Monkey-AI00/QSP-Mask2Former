"""Argument parser migration entry (behavior-compatible)."""

from __future__ import annotations

import argparse

import torch

import mecheye_live_pointrend_pointcloud_shape_prior as live_utils


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mech-Eye 在线抓取闭环：分割->目标点云->ICP->抓取位姿->机械臂执行")
    p.add_argument("--model-family", choices=["pointrend", "mask2former"], default="mask2former")
    p.add_argument("--config-file", default=live_utils._DEFAULT_POINTREND_CONFIG)
    p.add_argument("--config-file-prior", default=live_utils._DEFAULT_MASK2FORMER_QSP_CONFIG)
    p.add_argument("--mask2former-root", default=live_utils._DEFAULT_MASK2FORMER_ROOT)
    p.add_argument("--weights-prior", required=True, help="prior/QSP 权重")
    p.add_argument("--shape-prior-npy", default="")
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-classes", type=int, default=1)

    p.add_argument("--mask-mode", choices=["union", "maxscore"], default="union")
    p.add_argument("--pc-mask-mode", choices=["union", "maxscore", "iou"], default="iou")
    p.add_argument("--pc-iou-thresh", type=float, default=0.1)
    p.add_argument("--pc-join-dilate", type=int, default=25)
    p.add_argument("--mask-close", type=int, default=5)
    p.add_argument("--mask-dilate", type=int, default=0)
    p.add_argument("--mask-erode", type=int, default=0)
    p.add_argument("--invert-mask", action="store_true")
    p.add_argument("--auto-invert-mask", action="store_true")

    p.add_argument("--ip", default="", help="Mech-Eye 相机 IP")
    p.add_argument("--serial", default="", help="Mech-Eye 序列号")
    p.add_argument("--index", type=int, default=-1, help="discover index")
    p.add_argument("--exposure-seq", default="5,10")
    p.add_argument("--pc-smoothing", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-noise", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-outlier", choices=["", "off", "weak", "normal", "strong"], default="weak")
    p.add_argument("--pc-edge", choices=["", "sharp", "normal", "smooth"], default="normal")
    p.add_argument("--save-userset", action="store_true")

    p.add_argument("--cleargrasp", action="store_true")
    p.add_argument("--cleargrasp-normals-weights", default="")
    p.add_argument("--cleargrasp-outlines-weights", default="")
    p.add_argument("--cleargrasp-depth2depth-exe", default="")
    p.add_argument("--cleargrasp-out-w", type=int, default=256)
    p.add_argument("--cleargrasp-out-h", type=int, default=144)
    p.add_argument("--cleargrasp-inertia", type=float, default=1000.0)
    p.add_argument("--cleargrasp-smoothness", type=float, default=0.0001)
    p.add_argument("--cleargrasp-tangent", type=float, default=1.0)
    p.add_argument("--cleargrasp-fill-thresh", type=float, default=0.0)
    p.add_argument("--cleargrasp-filter-d", type=int, default=0)
    p.add_argument("--cleargrasp-filter-sigma-color", type=float, default=5.0)
    p.add_argument("--cleargrasp-filter-sigma-space", type=float, default=10.0)
    p.add_argument("--depth-unit", choices=["mm", "m"], default="mm")

    p.add_argument("--pc-stride", type=int, default=1)
    p.add_argument("--min-points", type=int, default=500)
    p.add_argument("--pp-voxel", type=float, default=0.0, help="目标点云预处理体素大小（米）")
    p.add_argument("--pp-sor-nb", type=int, default=50)
    p.add_argument("--pp-sor-std", type=float, default=1.0)
    p.add_argument("--pp-ror-nb", type=int, default=0)
    p.add_argument("--pp-ror-radius", type=float, default=0.0)
    p.add_argument("--pp-dbscan-eps", type=float, default=0.006, help="米")
    p.add_argument("--pp-dbscan-min-points", type=int, default=30)
    p.add_argument("--pp-keep-top-k", type=int, default=1)
    p.add_argument(
        "--pp-cluster-select",
        choices=["largest", "closest_z", "farthest_z", "smallest_bbox", "largest_bbox"],
        default="largest",
    )

    p.add_argument("--cad-ply", default="", help="局部抓取区域 CAD 模板点云（ICP 模式必填）")
    p.add_argument(
        "--cad-is-grasp-region-template",
        action="store_true",
        help="兼容旧参数：当前版本始终将 --cad-ply 视为局部模板，此开关可保留但不再影响逻辑。",
    )
    p.add_argument(
        "--grasp-region-preserve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="局部模板模式下保留抓取区域点云结构（默认 true，避免 DBSCAN largest-cluster 破坏局部几何）",
    )
    p.add_argument(
        "--grasp-region-init-json",
        default="",
        help="局部模板配准初始位姿 json（T_source_to_target/T_init/T_source_to_grasp_region_cad）",
    )
    p.add_argument(
        "--grasp-region-init-T",
        nargs=16,
        type=float,
        default=None,
        help="局部模板配准初始位姿 T_source_to_target（16个数，行优先）",
    )
    p.add_argument(
        "--grasp-region-to-grasp-json",
        default="",
        help="固定抓取参考变换 json；按当前脚本记号右乘使用：T_grasp_cam = T_region_cam @ T_region_to_grasp",
    )
    p.add_argument(
        "--T-grasp-region-to-grasp",
        nargs=16,
        type=float,
        default=None,
        help="固定抓取参考变换 T_region_to_grasp（16个数，行优先）",
    )
    p.add_argument("--icp-voxel", type=float, default=0.003, help="米（融合后 source 与局部模板 target 的 ICP 体素）")
    p.add_argument("--ransac-mult", type=float, default=1.5)
    p.add_argument("--icp-mult", type=float, default=0.7)
    p.add_argument(
        "--online-init-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用在线粗初始化候选（center/PCA/flip）",
    )
    p.add_argument("--init-candidate-max", type=int, default=12, help="在线初始化候选最大数量")
    p.add_argument("--coarse-icp-dist-mult", type=float, default=1.8, help="coarse ICP 对应距离系数（相对 icp_voxel）")
    p.add_argument("--refine-icp-dist-mult", type=float, default=0.7, help="refine ICP 对应距离系数（相对 icp_voxel）")
    p.add_argument(
        "--refine-fallback-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="当 refine 相对 coarse 明显劣化时，是否回退使用 coarse 结果",
    )
    p.add_argument("--refine-min-fitness-ratio", type=float, default=0.55, help="触发回退阈值：refine_fitness/coarse_fitness 下限")
    p.add_argument("--refine-max-residual-trans-mm", type=float, default=30.0, help="触发回退阈值：refine 残差平移上限（mm）")
    p.add_argument("--refine-max-residual-rot-deg", type=float, default=35.0, help="触发回退阈值：refine 残差旋转上限（deg）")
    p.add_argument(
        "--keep-coarse-track-on-refine-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refine 失败时是否保留 coarse 结果作为下一帧 warm start",
    )
    p.add_argument("--max-ransac", type=int, default=100000)
    p.add_argument("--max-icp", type=int, default=80, help="兼容参数：等价 stage2 最大迭代次数")
    p.add_argument("--max-icp-stage1", type=int, default=40, help="两阶段 ICP：stage1(point-to-point) 最大迭代")
    p.add_argument("--max-icp-stage2", type=int, default=80, help="两阶段 ICP：stage2(point-to-plane) 最大迭代")
    p.add_argument("--icp-stage1-mult", type=float, default=1.8, help="stage1 对应距离系数（相对 icp_dist）")
    p.add_argument("--icp-stage1-fitness-thr", type=float, default=0.20, help="stage1(point-to-point) fitness 下限")
    p.add_argument("--icp-stage1-rmse-thr", type=float, default=0.004, help="stage1(point-to-point) rmse 上限（米）")
    p.add_argument("--icp-coarse-fitness-thr", type=float, default=0.05, help="coarse_fitness 失败阈值，低于则跳过本帧")
    p.add_argument("--icp-fine-fitness-thr", type=float, default=0.15, help="fine_fitness 失败阈值，低于则跳过本帧")
    p.add_argument("--icp-rmse-thr", type=float, default=0.003, help="ICP inlier_rmse 失败阈值（米），高于则跳过本帧")
    p.add_argument("--bbox-ratio-min", type=float, default=0.60, help="source/target bbox extent 比例最小阈值（收紧默认）")
    p.add_argument("--bbox-ratio-max", type=float, default=1.60, help="source/target bbox extent 比例最大阈值（收紧默认）")
    p.add_argument("--surface-thin-enable", action=argparse.BooleanOptionalAction, default=True, help="是否启用局部点云主曲面提纯")
    p.add_argument("--surface-thin-band-mm", type=float, default=12.0, help="主曲面厚度带宽（mm）")
    p.add_argument("--surface-thin-axis", choices=["auto", "x", "y", "z"], default="auto", help="主曲面厚度方向（默认 auto）")
    p.add_argument("--surface-thin-min-points", type=int, default=500, help="主曲面提纯后最少点数")
    p.add_argument("--fusion-frames", type=int, default=6, help="局部点云融合缓存帧数 N")
    p.add_argument("--fusion-min-valid-frames", type=int, default=3, help="进入 ICP 前最少有效融合帧数")
    p.add_argument("--fusion-voxel", type=float, default=0.002, help="融合点云体素（米）")
    p.add_argument("--fused-min-points", type=int, default=600, help="融合后最少点数，低于则跳过本帧")
    p.add_argument(
        "--fusion-after-coarse-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="开启后先 coarse 对齐再写入融合缓存（缓解相机位姿变化导致的融合糊化）",
    )
    p.add_argument("--vis-icp", action="store_true", help="可视化 ICP 配准叠加结果（Open3D 阻塞窗口）")
    p.add_argument("--vis-icp-every", type=int, default=1, help="每 N 帧显示一次 ICP 叠加（默认每帧）")
    p.add_argument(
        "--vis-icp-only-fail",
        action="store_true",
        help="仅当 fine_fitness 低于阈值时显示 ICP 叠加（配合 --vis-icp-fail-thr）",
    )
    p.add_argument("--vis-icp-fail-thr", type=float, default=0.2, help="ICP 失败可视化阈值：fine_fitness <= 该值")

    p.add_argument(
        "--eye-in-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否按 Eye-in-Hand 动态计算 T_base_cam(t)。默认 true。",
    )
    p.add_argument(
        "--cam-flange-json",
        default="",
        help="Eye-in-Hand: T_cam_to_flange 4x4 json（相机坐标系 -> 法兰中心坐标系）；不填则使用脚本内置本次标定矩阵",
    )
    p.add_argument(
        "--T-cam-flange",
        nargs=16,
        type=float,
        default=None,
        help="Eye-in-Hand: 直接输入 T_cam_to_flange 4x4（16个数，行优先）",
    )
    p.add_argument("--base-cam-json", default="", help="Eye-to-Hand: 固定 T_base_to_camera 4x4 json")
    p.add_argument("--T-base-cam", nargs=16, type=float, default=None, help="Eye-to-Hand: 直接输入 T_base_to_camera 4x4")
    p.add_argument("--pregrasp-offset-mm", default="0,0,80")
    p.add_argument("--grasp-offset-mm", default="0,0,20")
    p.add_argument("--print-pose-json", action="store_true")

    p.add_argument("--auto-execute", action="store_true", help="自动下发机械臂动作")
    p.add_argument("--robot-ip", default="192.168.1.30")
    p.add_argument("--robot-user", type=int, default=0)
    p.add_argument("--robot-tool", type=int, default=0)
    p.add_argument("--robot-a", type=int, default=20)
    p.add_argument("--robot-v", type=int, default=20)
    p.add_argument("--robot-cp", type=int, default=0)
    p.add_argument(
        "--gripper-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否接入夹爪控制（RS485/Modbus RTU）。开启后支持到抓取点自动闭合、GUI 中按 p 回到p1，按 o 打开/按 c 关闭。",
    )
    p.add_argument("--gripper-port", default="/dev/ttyUSB0", help="夹爪串口")
    p.add_argument("--gripper-baudrate", type=int, default=115200, help="夹爪串口波特率")
    p.add_argument("--gripper-open-position", type=int, default=900, help="夹爪张开位置")
    p.add_argument("--gripper-close-position", type=int, default=0, help="夹爪闭合位置")
    p.add_argument("--gripper-init-timeout", type=float, default=5.0, help="夹爪初始化超时（秒）")
    p.add_argument("--gripper-feedback-interval", type=float, default=0.5, help="实时打印夹爪状态间隔（秒，<=0 关闭）")
    p.add_argument("--gripper-close-feedback-timeout", type=float, default=2.0, help="到抓取点闭合后，在抓取点等待并回传夹爪状态的时长（秒）")
    p.add_argument(
        "--final-pose-preset-retreat-mm",
        type=float,
        default=10.0,
        help="兼容旧参数：回到最终6D位姿前的 z 回退量（mm，默认 10）；若提供 --final-pose-preset-offset-mm，则本参数忽略",
    )
    p.add_argument(
        "--final-pose-preset-offset-mm",
        default="",
        help="预设点相对最终6D位姿的平移偏移 dx,dy,dz（mm），例如 0,-100,0 表示沿 y 负方向平移 100mm",
    )
    p.add_argument("--debug-use-fixed-rpy", action="store_true", help="IK 诊断：保持 xyz 不变，临时替换为固定 rpy 再做逆解")
    p.add_argument("--debug-fixed-rpy", default="64.5293,79.9632,1.9803", help="固定姿态角 rx,ry,rz（deg，逗号分隔）")
    p.add_argument(
        "--ik-candidate-rpy-json",
        default="",
        help="候选法兰姿态 JSON 文件（包含 candidates/ik_candidates/rpy_list 键，每项 [rx,ry,rz] deg）",
    )
    p.add_argument(
        "--ik-candidate-rpy-list",
        default="",
        help="候选法兰姿态列表（分号分隔多组，逗号分隔 rx,ry,rz；例 '64.5,80.0,2.0;-118.0,-2.8,8.6'）",
    )
    p.add_argument(
        "--ik-candidate-source-type",
        choices=["euler_rpy", "joint_guess"],
        default="euler_rpy",
        help="候选姿态来源类型：euler_rpy=TCP姿态角(rx,ry,rz)；joint_guess(J4/J5/J6)将被拒绝。",
    )
    p.add_argument(
        "--ik-runtime-mode",
        choices=["multi_rpy_search", "fixed_template_probe_execute"],
        default="multi_rpy_search",
        help="IK运行模式：multi_rpy_search(调试) / fixed_template_probe_execute(运行)。",
    )
    p.add_argument(
        "--fixed-template-rpy",
        default="64.5293,79.9632,1.9803",
        help="固定模板TCP姿态角 RX/RY/RZ（deg，运行模式使用）。",
    )
    p.add_argument(
        "--fixed-template-source",
        choices=["cli", "current_robot_pose", "json"],
        default="cli",
        help="固定模板姿态来源：命令行/当前机器人姿态/json。",
    )
    p.add_argument(
        "--fixed-template-json",
        default="",
        help="fixed-template-source=json 时使用，键支持 fixed_template_rpy/template_rpy/rpy。",
    )
    p.add_argument(
        "--runtime-probe-grasp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="运行模式下是否对 grasp 也做 motion probe（默认仅 probe pregrasp）。",
    )
    p.add_argument(
        "--ik-check-mode",
        choices=["api_only", "api_then_motion_probe", "motion_probe_only"],
        default="api_only",
        help="IK可达性检查模式：仅API / API失败后运动探测 / 仅运动探测。",
    )
    p.add_argument(
        "--motion-probe-execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否真正执行运动探测（默认 false，为安全仅dry-run）。",
    )
    p.add_argument(
        "--motion-probe-command",
        choices=["movj_pose", "movl_pose"],
        default="movj_pose",
        help="运动探测命令类型。",
    )
    p.add_argument(
        "--allow-motion-probe-success-as-reachable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="API失败但运动探测成功时，是否允许判定为可达。",
    )
    p.add_argument(
        "--prefer-api-ik-joint-execution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="有IK关节解时优先使用 movj_joint 执行。",
    )
    p.add_argument(
        "--direct-pose-fallback-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="无关节解但运动探测成功时，允许回退到笛卡尔直达执行。",
    )
    p.add_argument(
        "--ik-expand-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在 seed 候选姿态周围自动扩增搜索",
    )
    p.add_argument("--ik-expand-roll-deltas", default="-15,0,15", help="IK 自动扩增：roll 扰动集合（deg，逗号分隔）")
    p.add_argument("--ik-expand-pitch-deltas", default="-15,0,15", help="IK 自动扩增：pitch 扰动集合（deg，逗号分隔）")
    p.add_argument("--ik-expand-yaw-deltas", default="-30,0,30", help="IK 自动扩增：yaw 扰动集合（deg，逗号分隔）")
    p.add_argument("--ik-expand-max-candidates", type=int, default=64, help="IK 自动扩增后最大候选数")
    p.add_argument("--ik-expand-dedup-thr-deg", type=float, default=2.0, help="IK 扩增候选去重阈值（deg）")
    p.add_argument(
        "--ik-debug-fixed-rpy-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用固定 rpy 可达性诊断",
    )
    p.add_argument("--ik-debug-fixed-rpy", default="64.5293,79.9632,1.9803", help="固定 rpy 诊断姿态（deg，rx,ry,rz）")
    p.add_argument(
        "--ik-debug-run-on-fail-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅当候选 IK 全失败时运行固定 rpy 诊断",
    )
    p.add_argument("--stable-frames", type=int, default=3, help="候选 IK 模式：连续多少帧稳定后才执行抓取")
    p.add_argument("--stable-pos-thr-mm", type=float, default=5.0, help="稳定帧位置偏差阈值（mm）")
    p.add_argument("--stable-rot-thr-deg", type=float, default=10.0, help="稳定帧姿态偏差阈值（deg）")
    p.add_argument("--icp-fallback-after", type=int, default=5, help="ICP 连续失败多少帧后回退到人工初值")
    p.add_argument("--plan-before-recog", action="store_true", help="在识别前先执行一次机械臂规划到固定识别位")
    p.add_argument("--p1", default="-19.1160,26.4384,-122.4939,71.5494,85.4363,4.4136", help="识别前/抓取后回退固定关节位 p1（6轴角度，逗号分隔）")
    p.add_argument("--p2", default="103.4879,-33.7451,-76.8104,108.5125,88.4956,0.6603", help="抓取流程中 p1 后续到达的固定关节位 p2（6轴角度，逗号分隔）")
    p.add_argument("--handle-target-sample", action="store_true", help="调试模式：保存一帧局部抓取区域点云样本后退出（不进入ICP/机器人执行）")
    p.add_argument(
        "--handle-target-sample-path",
        default="/home/user/sjw/workspace/tmp/live_grasp_region_sample.ply",
        help="局部抓取区域样本点云输出路径（默认 /home/user/sjw/workspace/tmp/live_grasp_region_sample.ply）",
    )
    p.add_argument("--handle-target-sample-min-points", type=int, default=300, help="样本保存最少点数阈值")
    p.add_argument(
        "--source",
        default="",
        help="兼容参数：等价于 --handle-target-sample-path（可配合 --handle-target-sample 使用）",
    )

    p.add_argument("--win", default="Online Grasp Pipeline")
    p.add_argument("--wait", type=int, default=1)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--max-loops", type=int, default=0, help="仅 no-gui 时生效，>0 则循环指定次数后退出")

    # ---- 位姿估计后端选择（双环境解耦） ----
    p.add_argument(
        "--pose-backend",
        choices=["icp", "foundationpose"],
        default="icp",
        help="位姿估计后端：icp(默认，原有 ICP 链路) / foundationpose(子进程桥接)",
    )
    p.add_argument("--fp-conda-env", default="foundationpose", help="FoundationPose conda 环境名称")
    p.add_argument("--fp-bridge-dir", default="", help="runtime_bridge 目录（默认自动推断）")
    p.add_argument("--fp-script-path", default="", help="FP 桥接脚本路径（默认自动推断）")
    p.add_argument("--fp-code-dir", default="", help="FoundationPose 代码根目录（默认自动推断）")
    p.add_argument("--fp-mesh-file", default="", help="CAD 网格文件 (.obj/.stl)，FoundationPose 模式必填")
    p.add_argument("--fp-est-refine-iter", type=int, default=5, help="FoundationPose register 迭代次数")
    p.add_argument("--fp-min-n-views", type=int, default=20, help="FoundationPose 初始旋转网格视角数（减小可降显存）")
    p.add_argument("--fp-inplane-step", type=int, default=120, help="FoundationPose 初始旋转网格面内角步长（增大可降显存）")
    p.add_argument("--fp-timeout", type=int, default=120, help="子进程超时秒数")
    p.add_argument("--fp-debug", type=int, default=0, help="FoundationPose 内部 debug 级别")
    p.add_argument("--fp-debug-dir", default="/tmp/fp_debug", help="FP debug 输出目录")
    p.add_argument("--fp-region-in-obj-json", default="", help="T_region_in_obj JSON (整物体系→局部模板系)")
    p.add_argument("--T-fp-region-in-obj", nargs=16, type=float, default=None, help="T_region_in_obj 4×4（16 数，行优先）")

    return p


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


__all__ = ["build_argparser", "parse_args"]

