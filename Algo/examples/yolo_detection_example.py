# -*- coding: utf-8 -*-
"""在PyCharm直接运行的目标检测例程：输入图片，显示并保存结果。"""

import sys
from pathlib import Path

import cv2
import numpy as np

# 允许直接运行本文件，不要求设置PyCharm的工作目录。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algo.yolo import (YoloTask, YoloDevice, YoloStatus, PreprocessMode,
                       YoloModelConfig, YoloInferenceConfig, PreprocessConfig, load_model, infer)
from Algo.examples.yolo_example_output import save_and_show


# ==================== 修改这里后直接运行 ====================
MODEL_PATH = r"D:\models\detection.onnx"  # 替换为自己的目标检测模型，可使用PT或ONNX。
IMAGE_PATH = r"D:\images\part.png"  # 替换为自己的8位灰度/BGR图片。
OUTPUT_DIR = None  # None表示保存到原图旁的yolo_detection_results。
SHOW_WINDOW = True  # 显示原图与结果；False时只保存文件。

MODEL_CONFIG = YoloModelConfig(
    task=YoloTask.DETECT,
    device=YoloDevice.CPU,  # 可选AUTO或CUDA；CUDA时使用gpu_index。
    gpu_index=0,
)
INFERENCE_CONFIG = YoloInferenceConfig(
    preprocess=PreprocessConfig(
        mode=PreprocessMode.RESIZE_PADDING,  # 也可选PADDING或NONE。
        target_size=640,  # 与模型导出尺寸匹配。
    ),
    confidence_threshold=0.25,
    iou_threshold=0.45,
    merge_same_label=False,  # 显示原始框，分割时保持框和轮廓对应。
    include_segments=False,
)
# ===========================================================


def main():
    # 读图属于调用方示例，YOLO算子收到的始终是内存数组。
    image_path = Path(IMAGE_PATH).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"请修改顶部IMAGE_PATH，图片不存在：{image_path}")
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    if image is None:
        raise ValueError("无法解码输入图片，请检查文件内容。")

    # 模型只需加载一次，后续处理其他图像时可重复调用infer。
    model = load_model(MODEL_PATH, MODEL_CONFIG)
    if model.status is not YoloStatus.OK:
        raise RuntimeError(model.message)
    result = infer(model, image, INFERENCE_CONFIG)
    if result.status is not YoloStatus.OK:
        raise RuntimeError(result.message)

    # 零检测也是正常结果，仍保存和显示原图，并打印目标数量。
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else image_path.parent / "yolo_detection_results"
    return save_and_show(image, result, image_path, output_dir, SHOW_WINDOW)


if __name__ == "__main__":
    main()
