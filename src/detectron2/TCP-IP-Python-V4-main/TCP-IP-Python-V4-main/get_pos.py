import re
from dobot_api import DobotApiDashboard

robot_ip = "192.168.5.2"
dashboard_port = 29999

num_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def connect(robot_ip: str, dashboard_port: int):
    try:
        dashboard = DobotApiDashboard(robot_ip, dashboard_port)
        print(f"Successfully connected / 成功连接: {robot_ip}:{dashboard_port}")
        return dashboard
    except Exception as e:
        print(f"Failed to connect / 连接失败: {e}")
        return None


def parse_1_plus_6(resp: str):

    if resp is None:
        return None, None
    if "Not Tcp" in resp:
        print("Control Mode Is Not Tcp（请切到 TCP 控制模式）")
        return None, None

    nums = [float(x) for x in re.findall(num_pattern, resp)]
    if len(nums) < 7:
        return None, None

    err_id = int(nums[0])
    values = nums[1:7]
    return err_id, values


def main():
    dashboard = connect(robot_ip, dashboard_port)
    if dashboard is None:
        return

    resp_pose = dashboard.GetPose()
    print("Raw GetPose resp:", resp_pose)

    resp_angle = dashboard.GetAngle()
    print("Raw GetAngle resp:", resp_angle)

    pose_err, pose_xyzrxryrz = parse_1_plus_6(resp_pose)
    angle_err, joints_j1j6 = parse_1_plus_6(resp_angle)

    print("\nParsed:")
    print("  GetPose  ErrorID:", pose_err, "Pose[X,Y,Z,Rx,Ry,Rz]:", pose_xyzrxryrz)
    print("  GetAngle ErrorID:", angle_err, "Joints[J1..J6]:", joints_j1j6)


if __name__ == "__main__":
    main()