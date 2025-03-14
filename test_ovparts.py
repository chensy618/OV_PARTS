# import cv2
# import torch
# import numpy as np
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.transforms import ResizeShortestEdge
# from baselines import add_mask_former_config
# from detectron2.projects.deeplab import add_deeplab_config
# from baselines.data import SemanticObjPartDatasetMapper

# def setup_cfg(config_file, model_weights):
#     cfg = get_cfg()
#     add_deeplab_config(cfg)
#     add_mask_former_config(cfg)
#     cfg.merge_from_file(config_file)
#     cfg.MODEL.WEIGHTS = model_weights
#     cfg.freeze()
#     return cfg

# def preprocess_image(image_path, cfg):
#     # 读取图像
#     image = cv2.imread(image_path)
#     assert image is not None, f"无法读取图像 {image_path}"
#     # 转换为RGB格式
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # 获取输入格式
#     input_format = cfg.INPUT.FORMAT
#     # 获取图像高度和宽度
#     height, width = image.shape[:2]
#     # 创建数据增强
#     transform_gen = ResizeShortestEdge(
#         [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
#     )
#     # 应用数据增强
#     image = transform_gen.get_transform(image).apply_image(image)
#     # 转换为张量
#     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#     # 标准化
#     image = (image - torch.tensor(cfg.MODEL.PIXEL_MEAN)[:, None, None]) / torch.tensor(cfg.MODEL.PIXEL_STD)[:, None, None]
#     return {"image": image, "height": height, "width": width, "input_format": input_format}

# def main():
#     config_file = "/mnt/d/Github/OV_PARTS/configs/zero_shot/clipseg_ade.yaml"  # 配置文件路径
#     model_weights = "/mnt/d/Github/OV_PARTS/checkpoints/clipseg_ft_VA_L_F_voc.pth"  # 预训练模型权重路径
#     image_path = "/mnt/d/Github/OV_PARTS/CUB-200-2011/images/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0011_21820.jpg"     # 输入图像路径

#     # 设置配置
#     cfg = setup_cfg(config_file, model_weights)

#     # 构建模型
#     model = build_model(cfg)
#     model.eval()

#     # 加载模型权重
#     checkpointer = DetectionCheckpointer(model)
#     checkpointer.load(cfg.MODEL.WEIGHTS)

#     # 预处理图像
#     inputs = preprocess_image(image_path, cfg)
#     inputs = {"image": inputs["image"].to(cfg.MODEL.DEVICE)}

#     # 推理
#     with torch.no_grad():
#         outputs = model([inputs])[0]

#     # 获取元数据
#     metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

#     # 可视化结果
#     image = cv2.imread(image_path)
#     visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
#     if "sem_seg" in outputs:
#         sem_seg = outputs["sem_seg"].argmax(dim=0).to(torch.device("cpu"))
#         vis_output = visualizer.draw_sem_seg(sem_seg)
#     else:
#         raise NotImplementedError("模型输出不包含语义分割结果。")

#     # 显示结果
#     cv2.imshow("Segmentation", vis_output.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data.transforms import ResizeShortestEdge
from baselines import add_mask_former_config
from detectron2.projects.deeplab import add_deeplab_config


def setup_cfg(config_file, model_weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.freeze()
    return cfg


def preprocess_image(image_path, cfg):
    image = cv2.imread(image_path)
    assert image is not None, f"Cannot read image {image_path}"

    original_height, original_width = image.shape[:2]

    transform_gen = ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = transform_gen.get_transform(image_rgb).apply_image(image_rgb)

    image_tensor = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
    image_tensor = (image_tensor - torch.tensor(cfg.MODEL.PIXEL_MEAN)[:, None, None]) / torch.tensor(cfg.MODEL.PIXEL_STD)[:, None, None]

    inputs = {
        "image": image_tensor,
        "height": original_height,
        "width": original_width,
        "file_name": image_path
    }
    return inputs


def main():
    config_file = "/mnt/d/Github/OV_PARTS/configs/zero_shot/clipseg_ade.yaml"
    model_weights = "/mnt/d/Github/OV_PARTS/checkpoints/clipseg_ft_VA_L_F_voc.pth"
    image_path = "/mnt/d/Github/OV_PARTS/CUB-200-2011/images/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0011_21820.jpg"

    cfg = setup_cfg(config_file, model_weights)

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    inputs = preprocess_image(image_path, cfg)

    # 手动指定你感兴趣的类别索引
    inputs_device = {
        "image": inputs["image"].to(cfg.MODEL.DEVICE), 
        "file_name": inputs["file_name"],
        "sem_seg": torch.tensor([1, 2, 3], device=cfg.MODEL.DEVICE)  # 示例索引
    }

    with torch.no_grad():
        outputs = model([inputs_device])[0]

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    original_image = cv2.imread(image_path)
    visualizer = Visualizer(original_image[:, :, ::-1], metadata=metadata, scale=1.0)

    if "sem_seg" in outputs:
        sem_seg = outputs["sem_seg"].argmax(dim=0).to("cpu")
        vis_output = visualizer.draw_sem_seg(sem_seg)
        result_image = vis_output.get_image()[:, :, ::-1]
    else:
        raise NotImplementedError("Model output does not contain semantic segmentation results.")

    cv2.imshow("Segmentation", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()