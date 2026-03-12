import pyrealsense2 as rs

# 1. 配置并开启 Pipeline
pipeline = rs.pipeline()
config = rs.config()

# 注意：分辨率不同，内参就不一样！务必与你实际使用的分辨率一致
W, H = 1280, 720
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

# 2. 启动流，获取 profile
profile = pipeline.start(config)

try:
    # 3. 获取特定流的内参 (以 Color 为例，因为 YOLO 是跑在 Color 上的)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    # 4. 打印内参信息
    print("=== Color Camera Intrinsics ===")
    print(f"Width: {intr.width}, Height: {intr.height}")
    print(f"PPX (cx): {intr.ppx}, PPY (cy): {intr.ppy}")
    print(f"FX (fx): {intr.fx}, FY (fy): {intr.fy}")
    print(f"Distortion: {intr.coeffs}") # 畸变系数
    print(f"Model: {intr.model}")

    # 获取 Depth 内参 (如果需要)
    depth_stream = profile.get_stream(rs.stream.depth)
    d_intr = depth_stream.as_video_stream_profile().get_intrinsics()
    print("\n=== Depth Camera Intrinsics ===")
    print(f"FX: {d_intr.fx}, FY: {d_intr.fy}")

finally:
    pipeline.stop()