# -*- coding: utf-8 -*-
"""模型加载、单图与批量推理统一入口，仅接收内存图像。"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .postprocess import merge_same_label_intersections, parse_result
from .preprocess import prepare_image
from .types import (YoloDevice, YoloInferenceConfig, YoloModelConfig, YoloResult,
                    YoloStatus, YoloTiming)


@dataclass
class YoloModel:
    """模型对象与加载状态，外部无需接触后端原始结果。"""
    model_path: str  # 本地模型文件路径，可重复加载定位问题。
    config: YoloModelConfig  # 模型任务和设备选择，不包含逐次推理阈值。
    model: object = None  # 后端模型实例，失败时为None；调用方通常无需直接操作。
    status: YoloStatus = YoloStatus.ERROR  # 加载状态；即使OK，后端仍可能在首次推理延迟初始化时报错。
    message: str = ""  # 加载成功说明或具体错误原因。


def load_model(model_path, config: Optional[YoloModelConfig] = None) -> YoloModel:
    """构造可重复推理的模型上下文，不在此处读取图片。

    model_path：本地模型路径；config：任务/设备配置，省略时使用默认配置。
    参数类型错误抛TypeError；文件缺失或加载失败返回status=ERROR及message。
    OK表示模型对象已建立；部分后端延迟到首次predict初始化，仍可能在推理时失败。
    """
    if config is None:
        config = YoloModelConfig()
    if not isinstance(config, YoloModelConfig):
        raise TypeError("config must be YoloModelConfig.")
    if not isinstance(model_path, (str, os.PathLike)):
        raise TypeError("model_path must be a local path.")
    path = Path(model_path).expanduser()
    context = YoloModel(str(path), config)
    if not path.is_file():
        context.message = "Model file does not exist: " + str(path)
        return context
    try:
        # 延迟导入，常规图像/几何操作无需安装Ultralytics；不主动安装依赖。
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO
        context.model = YOLO(str(path), task=config.task.value)
        # PT等模型可能自带任务信息，检查加载结果，避免用检测解析器处理其他任务。
        actual_task = getattr(context.model, "task", config.task.value)
        if actual_task != config.task.value:
            raise ValueError("model task does not match configured task.")
        context.status = YoloStatus.OK
        context.message = "YOLO model loaded."
    except Exception as exc:
        context.model = None
        context.message = "YOLO model load failed: " + str(exc)
    return context


def infer(context: YoloModel, image: np.ndarray,
          config: Optional[YoloInferenceConfig] = None) -> YoloResult:
    """推理一张uint8灰度/BGR内存图，返回YoloResult。

    context为load_model返回的上下文；config为可选推理配置，可逐次更换阈值。
    复用批量流程保证单图/批量预后处理一致；若image为上游ROI，输出也是ROI内坐标。
    """
    return infer_batch(context, [image], config)[0]


def infer_batch(context: YoloModel, images: Sequence[np.ndarray],
                config: Optional[YoloInferenceConfig] = None) -> List[YoloResult]:
    """一次predict处理图像列表/元组，按输入顺序返回等长的结果列表。

    每项必须是uint8灰度/BGR数组；空集合返回[]，不接受图片路径或生成器。
    非法输入抛异常；后端执行或结果解析失败时，整批各项都返回ERROR和原因。
    模型的导出batch维度决定实际批量能力；本函数不会自动拆批或回退到逐张推理。
    """
    if not isinstance(context, YoloModel):
        raise TypeError("context must be YoloModel.")
    if config is None:
        config = YoloInferenceConfig()
    if not isinstance(config, YoloInferenceConfig):
        raise TypeError("config must be YoloInferenceConfig.")
    if not isinstance(images, (list, tuple)):
        raise TypeError("images must be a list or tuple of numpy images.")
    if not images:
        return []
    # 先统一校验并预处理整批，防止路径字符串被后端当作文件读取。
    start = time.perf_counter()
    prepared = [prepare_image(image, config.preprocess) for image in images]
    preprocess_ms = (time.perf_counter() - start) * 1000
    if context.status is not YoloStatus.OK or context.model is None:
        return [YoloResult(YoloStatus.ERROR, context.message or "Model unavailable.",
                           context.model_path, config, transform=transform)
                for _, transform in prepared]
    sources = []
    for image, _ in prepared:
        # Ultralytics的NumPy入口约定BGR，其内部再转RGB；显式适配，避免重复换色。
        sources.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                       if config.preprocess.convert_bgr_to_rgb else image)
    # AUTO交给后端决策；CUDA使用卡号字符串。本库不在这里执行设备探测或失败回退。
    device = None if context.config.device is YoloDevice.AUTO else (
        "cpu" if context.config.device is YoloDevice.CPU else str(context.config.gpu_index))
    try:
        start = time.perf_counter()
        # imgsz是尺寸请求，后端仍可能按步长/固定输入调整；不是跳过其内部预处理。
        backend_results = context.model.predict(
            source=sources, conf=config.confidence_threshold, iou=config.iou_threshold,
            device=device, imgsz=config.preprocess.target_size, verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000
        if len(backend_results) != len(images):
            raise RuntimeError("batch result count mismatch.")
        responses = []
        for raw, (_, transform) in zip(backend_results, prepared):
            start = time.perf_counter()
            detections, segments = parse_result(raw, transform, config.include_segments)
            # 只合并检测框；segments仍对应原始检测，不能与merged按索引重新配对。
            merged = merge_same_label_intersections(detections) if config.merge_same_label else detections
            responses.append(YoloResult(
                status=YoloStatus.OK, message="YOLO inference finished.", model_path=context.model_path,
                config=config, detections=merged, segments=segments,
                raw_detections=detections if config.merge_same_label else [], transform=transform,
                # predict包含后端自己的预处理；这两项耗时是整批共享值，不能逐图求和。
                timing=YoloTiming(preprocess_ms, inference_ms, (time.perf_counter() - start) * 1000, len(images)),
            ))
        return responses
    except Exception as exc:
        # 不把解析失败的某张静默当成无目标，整批返回错误，避免调用方误用部分成功数据。
        return [YoloResult(YoloStatus.ERROR, "YOLO inference failed: " + str(exc),
                           context.model_path, config, transform=transform)
                for _, transform in prepared]
