# -*- coding: utf-8 -*-
"""模型结果解析、坐标还原和同类框交集合并，不执行图像预处理。"""

from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .types import ImageTransform, YoloBox, YoloDetection, YoloPoint, YoloSegment


def restore_box(box: YoloBox, transform: ImageTransform, clip: bool = True) -> YoloBox:
    """将YoloBox从预处理图还原到本次输入图，返回新的xywh矩形。

    transform来自prepare_image；clip=True裁剪到输入图边界，False保留越界坐标。
    输入图若为ROI，不加回上游裁剪偏移；本函数不修改box或transform。
    """
    if not isinstance(box, YoloBox) or not isinstance(transform, ImageTransform):
        raise TypeError("box/transform must be YoloBox/ImageTransform.")
    # 先消除补边平移，再分别除以横纵比例；先还原两角，再重新计算宽高。
    x1, y1 = (box.x - transform.pad_left) / transform.scale_x, (box.y - transform.pad_top) / transform.scale_y
    x2 = (box.x + box.width - transform.pad_left) / transform.scale_x
    y2 = (box.y + box.height - transform.pad_top) / transform.scale_y
    # 框使用连续几何边界，右/下边界允许等于width/height，而不是数组索引上限。
    if clip:
        x1, x2 = (max(0.0, min(v, transform.original_width)) for v in (x1, x2))
        y1, y2 = (max(0.0, min(v, transform.original_height)) for v in (y1, y2))
    return YoloBox(float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1)))


def restore_boxes(boxes: Iterable[YoloBox], transform: ImageTransform, clip: bool = True) -> List[YoloBox]:
    """批量还原矩形，空集合返回空列表。"""
    return [restore_box(box, transform, clip) for box in boxes]


def restore_polygon(points: Iterable, transform: ImageTransform, clip: bool = True) -> List[YoloPoint]:
    """将Nx2数组、坐标序列或YoloPoint列表还原为List[YoloPoint]。

    空集合返回[]；clip语义与restore_box一致。仅逐点变换，不重新排序、闭合或重采样。
    """
    if not isinstance(transform, ImageTransform):
        raise TypeError("transform must be ImageTransform.")
    restored = []
    for point in points:
        px, py = (point.x, point.y) if isinstance(point, YoloPoint) else point
        x = (float(px) - transform.pad_left) / transform.scale_x
        y = (float(py) - transform.pad_top) / transform.scale_y
        if clip:
            x, y = max(0.0, min(x, transform.original_width)), max(0.0, min(y, transform.original_height))
        restored.append(YoloPoint(float(x), float(y)))
    return restored


def parse_result(result, transform: Optional[ImageTransform] = None,
                 include_segments: bool = True) -> Tuple[List[YoloDetection], List[YoloSegment]]:
    """把后端单图结果转换成(detections, segments)两个结构体列表。

    result使用Ultralytics的boxes/names/masks接口；transform省略时保持后端坐标。
    include_segments=False跳过轮廓解析，不改变模型任务；无目标正常返回空列表。
    后端框坐标应已经映射回传入predict的图，不能传未经后端解码的网络原始张量。
    """
    if result is None:
        raise ValueError("backend returned no result object.")
    if transform is not None and not isinstance(transform, ImageTransform):
        raise TypeError("transform must be ImageTransform.")
    # Ultralytics的cpu()复制结果张量，兼容CPU及CUDA，不直接用NumPy读取CUDA张量。
    result = result.cpu() if callable(getattr(result, "cpu", None)) else result
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})
    detections, segments = [], []
    for item in boxes if boxes is not None else []:
        values = np.asarray(item.xyxy).reshape(-1)
        if values.size != 4:
            raise ValueError("invalid backend box shape.")
        x1, y1, x2, y2 = map(float, values)
        # 后端xyxy统一转换为库的xywh；保留预处理坐标框，方便排查坐标变换。
        pre_box = YoloBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))
        box = restore_box(pre_box, transform) if transform is not None else pre_box
        class_id = int(np.asarray(item.cls).reshape(-1)[0])
        confidence = float(np.asarray(item.conf).reshape(-1)[0])
        if isinstance(names, dict):
            label = str(names.get(class_id, class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            label = str(names[class_id])
        else:
            label = str(class_id)
        detections.append(YoloDetection(label, class_id, confidence, box, pre_box))
    # 分割沿用每个原始框的类别和置信度，polygon保存轮廓点，不生成整图像素Mask。
    if include_segments:
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if polygons is not None:
            # 不用数组的布尔值判断；轮廓与检测框必须一一对应，禁止静默错配类别。
            if len(polygons) != len(detections):
                raise ValueError("backend polygon/box count mismatch.")
            for detection, polygon in zip(detections, polygons):
                points = restore_polygon(polygon, transform) if transform is not None else [
                    YoloPoint(float(x), float(y)) for x, y in polygon]
                segments.append(YoloSegment(detection.label, detection.class_id, detection.confidence,
                                            detection.box, points))
    return detections, segments


def merge_same_label_intersections(detections: Iterable[YoloDetection]) -> List[YoloDetection]:
    """保持历史交集框规则，非NMS或并集；按置信度降序处理且不修改原对象。"""
    merged = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        box = detection.box
        for index, target in enumerate(merged):
            if target.label.strip().lower() != detection.label.strip().lower():
                continue
            other = target.box
            # 相交区域必须具有正面积；仅边界接触不合并，同类匹配沿用标签归一化规则。
            x1, y1 = max(box.x, other.x), max(box.y, other.y)
            x2 = min(box.x + box.width, other.x + other.width)
            y2 = min(box.y + box.height, other.y + other.height)
            if x2 <= x1 or y2 <= y1:
                continue
            # 当前target先进入列表，置信度不低于新框；只替换几何框并累计数量。
            # 多框时是按顺序贪心处理，不是传递闭包，后续比较使用已经缩小的交集框。
            merged[index] = replace(target, box=YoloBox(x1, y1, x2 - x1, y2 - y1),
                                    merged_count=target.merged_count + detection.merged_count)
            break
        else:
            merged.append(detection)
    return merged
