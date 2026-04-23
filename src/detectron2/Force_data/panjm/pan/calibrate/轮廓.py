# 轮廓提取
import cv2
import numpy as np
import os


def detect_contours_and_draw_rectangles(image_path,save_path,lable):
    # 读取图像
    image = cv2.imread(image_path)
    # print(image)
    original_image = image.copy()

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    # cv2.imwrite('blurred3.jpg', blurred)
    # 使用 Canny 算子检测边缘
    # edges = cv2.Canny(blurred, 50, 150)    # 能看见清晰的轮廓
    edges = cv2.Canny(blurred, 20, 70)    # 矩形框比较合适

    # 检测轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_points = []
    index = 1
    # 绘制轮廓和最小外接矩形
    for idx, contour in enumerate(contours):
        # 绘制轮廓
        cv2.drawContours(original_image, contours, idx, (0, 255, 0), 2)  # 绘制绿色轮廓

        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        width = int(rect[1][0])  # 这个是直观上的高
        height = int(rect[1][1])    # 这个是直观上的宽
        # if 160 <= width <= 180 and 200 <= height <= 500:
        if 0<= width  and 10 <= height :
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 绘制最小外接矩形
            cv2.drawContours(original_image, [box], 0, (255, 0, 0), 2)  # 绘制蓝色矩形

            # 计算矩形中心点
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])

            # center_points.append((center_x, center_y))

            # 标注序号和中心坐标
            cv2.putText(original_image, str(index), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA) 
            
            print(f"第{index}个矩形, 高: {width}, 宽: {height}, 中心: ({center_x}, {center_y})")
            index += 1
            

    # 保存结果图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_image_path = os.path.join(save_path, '{}.jpg'.format(lable)) 
    cv2.imwrite(output_image_path, original_image)

    # 可视化
    # cv2.imshow('Contours and Rectangles', original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':

    save_path = '/home/iqr/pan/calibrate/rectangles_select'
    for i in range(10,11):
        image_path = f'/home/iqr/pan/calibrate/image/{i}.jpg'
        print(image_path)
        print(f'-------{i}----------')
        detect_contours_and_draw_rectangles(image_path,save_path,i)