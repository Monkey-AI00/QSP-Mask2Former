import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
# 引入 PointRend 的配置加载器 (关键)
from point_rend import add_pointrend_config

# 1. 配置初始化
cfg = get_cfg()
add_pointrend_config(cfg) # 必须添加这一行，否则无法识别 PointRend 特有的配置项

# 2. 加载配置文件 (使用相对路径，假设你在 projects/PointRend 下运行)
cfg.merge_from_file("configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

# 3. 加载你刚才提供的权重路径 (绝对路径)
cfg.MODEL.WEIGHTS = "/home/users1/sjw/cursor/Yolo_pointrend/detectron2/weights/model_final_edd263.pkl"

# 4. 设置推理参数
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 置信度阈值
cfg.MODEL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 5. 创建推理器
print(f"Loading model from: {cfg.MODEL.WEIGHTS} ...")
predictor = DefaultPredictor(cfg)
print("Model loaded successfully!")

# 6. 读取一张图片进行测试 
# (这里我们先下载一张网图，或者你可以把路径改成你的线缆图片路径)
import urllib.request
url = 'http://images.cocodataset.org/val2017/000000439715.jpg' # 一张经典的骑马图
img_path = "input_2.jpg"
if not os.path.exists(img_path):
    print("Downloading sample image...")
    urllib.request.urlretrieve(url, img_path)

im = cv2.imread(img_path)

# 7. 执行推理
print("Running inference...")
outputs = predictor(im)
# 获取实例预测结果并转到 CPU
instances = outputs["instances"].to("cpu")
# 仅显示分割：去掉包络框，直接用掩膜可视化
if instances.has("pred_boxes"):
    instances.remove("pred_boxes")

# 8. 可视化结果（仅掩膜）
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
if instances.has("pred_masks"):
    out = v.overlay_instances(masks=instances.pred_masks)
else:
    # 若没有掩膜字段，则回退到默认的实例可视化
    out = v.draw_instance_predictions(instances)

# 9. 保存结果
output_filename = "output_pointrend.jpg"
cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])
print(f"Result saved to {output_filename}. Please check it!")