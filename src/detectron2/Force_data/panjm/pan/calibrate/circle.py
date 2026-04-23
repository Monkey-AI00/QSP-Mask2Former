import cv2
import numpy as np

def detect_circles(image):
    """
    检测图像中的圆，并返回圆心的像素坐标
    :param image: 输入图像
    :return: 圆心坐标列表 [(x1, y1), (x2, y2), ...]
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊处理，减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用 HoughCircles 方法检测圆
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, 
                               param1=100, param2=30, minRadius=150, maxRadius=200)

    circle_centers = []
    if circles is not None:
        circles = circles[0, :]
        print(circles[0][:1])
        circle_centers = [(x, y,r) for (x, y, r) in circles]
        circle_list = [(a,b) for (a,b,c) in circles]
        circle_list.sort(key=lambda c: (c[1], c[0]))
        circle_centers.sort(key=lambda c: (c[1], c[0]))  # “Z”字形排序

        # 遍历排序后的圆心坐标
        for idx, (x, y,r) in enumerate(circle_centers):
            # 画绿色圆圈
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # 绿色的圆
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 2)
            # 标注圆心坐标
            text = f"({x:.2f}, {y:.2f})"
            cv2.putText(image, text, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)  # 红色的小字体文本

    return circle_list, image


def calculate_affine_transform(pixel_coords, real_coords):
    """
    计算像素坐标与真实坐标之间的仿射变换矩阵
    :param pixel_coords: 像素坐标列表 [(x1, y1), (x2, y2), ...]
    :param real_coords: 真实坐标列表 [(X1, Y1), (X2, Y2), ...]
    :return: 仿射变换矩阵 (2x3)
    """
    pixel_coords = np.array(pixel_coords, dtype=np.float32)
    real_coords = np.array(real_coords, dtype=np.float32)

    # 计算仿射变换矩阵
    transform_matrix, _ = cv2.estimateAffine2D(pixel_coords, real_coords)
    return transform_matrix


def apply_affine_transform(matrix, pixel_point):
    pixel_point = np.array([pixel_point[0], pixel_point[1], 1], dtype=np.float32)  # 扩展为3D点
    real_point = np.dot(matrix, pixel_point)  # 应用仿射矩阵
    return real_point[:2]  # 返回x, y

image_path = './10_31.jpg'
image = cv2.imread(image_path)

# 检测圆并获取圆心坐标
centers, annotated_image = detect_circles(image)

# 输出圆心坐标
print("检测到的圆心坐标: ", centers)

# 可视化并保存结果
# cv2.imshow('Annotated Image', annotated_image)
cv2.imwrite('333333.jpg', annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

robot_pos = [(0.1,0.2),(0.15,0.2),(0.2,0.2),(0.1,0.25),(0.15,0.25),(0.2,0.25),(0.1,0.3),(0.15,0.3),(0.2,0.3)]
affine_matrix = calculate_affine_transform(centers, robot_pos)
print("仿射变换矩阵:\n", affine_matrix)

# # 示例新像素坐标
# new_pixel_coord = (130, 59.4)
# # 计算其真实坐标
# new_real_coord = apply_affine_transform(affine_matrix, new_pixel_coord)
# print("新像素坐标对应的真实坐标:", new_real_coord)

'''圆[(522.60004, 335.40002),(1217.4, 330.6),(1913.4, 330.6),
    (527.4, 1030.2001),(1225.8, 1026.6001),(1920.6001, 1020.60004),
    (537.0, 1725.0001), (1228.2001, 1717.8),(1925.4, 1711.8)]'''