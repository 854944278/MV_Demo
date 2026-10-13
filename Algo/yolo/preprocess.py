# -*- coding: utf-8 -*-
"""只负责输入图像的缩放、补边和通道整理，不读文件、不还原坐标。"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .types import ImageTransform, PreprocessConfig, PreprocessMode


def prepare_image(image: np.ndarray, config: Optional[PreprocessConfig] = None) -> Tuple[np.ndarray, ImageTransform]:
    """准备模型输入，返回(processed_image, ImageTransform)。

    image为非空uint8灰度/BGR数组；config省略时使用PreprocessConfig默认值。
    输出为独立三通道数组，默认BGR，可选RGB；NONE只跳过本库缩放补边。
    输入类型/模式不合法会报错；坐标还原由postprocess负责，不在这里执行。
    """
    if config is None:
        config = PreprocessConfig()
    if not isinstance(config, PreprocessConfig):
        raise TypeError("config must be PreprocessConfig.")
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3))):
        raise ValueError("image must be a non-empty uint8 grayscale or BGR array.")
    height, width = image.shape[:2]
    # 灰度统一复制为BGR三通道；始终返回独立数组，后端不能改动上游图像。
    source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    left = top = right = bottom = 0
    resized_width, resized_height = width, height
    if config.mode is not PreprocessMode.NONE:
        target = config.target_size
        if config.mode is PreprocessMode.PADDING:
            if width > target or height > target:
                raise ValueError("padding mode cannot shrink an image larger than target_size.")
        else:
            # 长边缩到target，短边按同比例取整；至少保留1像素以支持细长ROI。
            scale = target / float(max(width, height))
            resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
            # 缩小使用面积插值减少混叠，放大使用线性插值；变换记录保存实际取整尺寸。
            source = cv2.resize(source, (resized_width, resized_height),
                                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        pad_width, pad_height = target - resized_width, target - resized_height
        # 居中补边；奇数余量分给右/下，因此还原必须使用实际left/top，不能假定完全对称。
        left, top = pad_width // 2, pad_height // 2
        right, bottom = pad_width - left, pad_height - top
        source = cv2.copyMakeBorder(source, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=config.padding_color)
    # 补边颜色按BGR解释，先补边再整体换色，避免图像和边框通道顺序不一致。
    if config.convert_bgr_to_rgb:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    transform = ImageTransform(width, height, resized_width, resized_height, left, top, right, bottom)
    return source, transform
