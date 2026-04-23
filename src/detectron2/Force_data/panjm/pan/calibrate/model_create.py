import cv2
import numpy as np

# 定义全局变量来保存选定的区域
start_point = None
end_point = None
drawing = False

# 鼠标回调函数
def select_template(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

# 加载图像
image_path = '/home/iqr/pan/calibrate/image2/1.jpg'
image = cv2.imread(image_path)

# 创建窗口并设置鼠标回调
cv2.namedWindow('Select Template')
cv2.setMouseCallback('Select Template', select_template)

while True:
    # 显示图像
    temp_image = image.copy()

    # 如果选择的区域存在，则绘制矩形、外接圆和外接矩形
    if start_point and end_point:
        cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)  # 手动框出的矩形

        # 计算外接圆的中心和半径
        x1, y1 = start_point
        x2, y2 = end_point
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # 计算外接圆的半径（矩形对角线的一半）
        diagonal_length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        radius = diagonal_length // 2

        # 绘制外接圆
        cv2.circle(temp_image, center, radius, (255, 0, 0), 2)  # 外接圆（蓝色）

        # 计算外接矩形的四个顶点
        rect_x1 = center[0] - radius
        rect_y1 = center[1] - radius
        rect_x2 = center[0] + radius
        rect_y2 = center[1] + radius

        # 绘制外接矩形
        cv2.rectangle(temp_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), 2)  # 外接矩形（红色）

    cv2.imshow('Select Template', temp_image)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 确保选择的区域有效
if start_point and end_point:
    # 使用外接矩形的坐标生成模板
    template = image[rect_y1:rect_y2, rect_x1:rect_x2]  # 生成模板
    cv2.imwrite('/home/iqr/pan/calibrate/image2/3_model2.jpg', template)  # 保存模板
    print("Template saved as 'template_with_circle_and_rect.jpg'.")

# 关闭窗口
cv2.destroyAllWindows()

