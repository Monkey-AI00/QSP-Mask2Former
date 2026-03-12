import pyrealsense2 as rs

# 1. 初始化 Pipeline 和 对齐器
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 【重要】创建对齐对象：让深度图向彩色图对齐
align_to = rs.stream.color
align = rs.align(align_to)

profile = pipeline.start(config)

try:
    # 假设这是 YOLO 识别到的中心点像素坐标
    u_center = 992  # x
    v_center = 518  # y
    # 1. 初始化滤镜
    temporal = rs.temporal_filter()
    spatial = rs.spatial_filter()  # 空间滤镜，填补空洞

    while True:
        frames = pipeline.wait_for_frames()

        # 2. 【关键】执行对齐
        aligned_frames = align.process(frames)

        # 3. 获取对齐后的深度帧
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame: continue

        # 4. 直接读取深度 Z (API 会自动把原始值转为米)
        # 参数顺序: get_distance(x, y) -> get_distance(u, v)
        z_value = depth_frame.get_distance(u_center, v_center)

        if z_value == 0:
            print("警告：该点位于盲区或无效，深度为 0")
        else:
            print(f"坐标 ({u_center}, {v_center}) 的距离 Z = {z_value:.3f} 米")

finally:
    pipeline.stop()