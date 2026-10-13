# -*- coding: utf-8 -*-
"""将输入图像均分为条带，平均降维后按扫描方向查找单边缘或边缘对。"""

from typing import Optional

import cv2
import numpy as np

from .types import (CaliperConfig, CaliperResult, CaliperSample, Edge, EdgeMode,
                    EdgePair, EdgePolarity, EdgeSelection, PairPolarity, Point, ScanDirection)


def _profile_edges(profile, horizontal, reverse, center):
    """提取梯度幅值的局部峰；等高平台只返回中点，坐标还原到输入图像。"""
    # profile已按搜索方向排列；差分为正表示暗到亮，为负表示亮到暗。
    # horizontal决定坐标落在哪个轴，center是另一轴上的条带中心坐标。
    gradient = np.diff(profile)
    magnitude = np.abs(gradient)
    edges = []
    index = 0
    while index < len(gradient):
        end = index
        # 连续等值且同极性的梯度属于同一个平台，避免缓坡产生重复边缘。
        while end + 1 < len(gradient) and gradient[end + 1] == gradient[index]:
            end += 1
        strength = float(magnitude[index])
        # 图像两端没有外侧邻居，用-1表示缺失比较值，不虚构边界外的灰度。
        left = float(magnitude[index - 1]) if index else -1.0
        right = float(magnitude[end + 1]) if end + 1 < len(gradient) else -1.0
        # 相反极性的响应分别比较，保留窄亮条/暗条的两侧边缘。
        if index and gradient[index - 1] * gradient[index] < 0:
            left = -1.0
        if end + 1 < len(gradient) and gradient[end + 1] * gradient[index] < 0:
            right = -1.0
        if strength > 0 and strength > left and strength > right:
            # 差分发生在两像素中心之间，故加0.5；平台取中点，不是亚像素拟合。
            # 反向时position仍沿扫描方向递增，coordinate则还原为图像轴坐标。
            position = (index + end) / 2.0 + 0.5
            coordinate = len(profile) - 1 - position if reverse else position
            edges.append(Edge(
                x=float(coordinate if horizontal else center),
                y=float(center if horizontal else coordinate),
                position=float(position), strength=strength,
                polarity=EdgePolarity.DARK_TO_BRIGHT if gradient[index] > 0 else EdgePolarity.BRIGHT_TO_DARK,
            ))
        index = end + 1
    return edges


def caliper_find_edges(image: np.ndarray, config: Optional[CaliperConfig] = None) -> CaliperResult:
    """在输入图像的各条带内独立找边，不拟合直线、不判产品合格。

    image：uint8灰度(H,W)或BGR(H,W,3)数组，整个输入即检测区域，不修改原数组。
    config：CaliperConfig；省略时使用默认值，详细字段说明见types.py。
    返回CaliperResult：保留全部采样，包括未找到目标的条带；坐标相对输入图。
    类型错误抛TypeError，图像/采样尺寸不合法抛ValueError；无边缘正常返回found=False。
    """
    if config is None:
        config = CaliperConfig()
    if not isinstance(config, CaliperConfig):
        raise TypeError("config must be CaliperConfig.")
    # 配置已在构造时校验且不可变，此处只检查与实际图像有关的约束。
    direction, edge_mode, selection = config.direction, config.edge_mode, config.selection
    sample_count, min_contrast = config.sample_count, config.min_contrast
    smooth_sigma, min_width, max_width = config.smooth_sigma, config.min_width, config.max_width
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3))):
        raise ValueError("image must be a non-empty uint8 grayscale or BGR array.")
    # 左右搜索沿X轴，但分区沿Y轴；上下搜索沿Y轴，但分区沿X轴。
    # reverse只反转每条剖面的扫描顺序，不反转条带编号。
    horizontal = direction in (ScanDirection.LEFT_TO_RIGHT, ScanDirection.RIGHT_TO_LEFT)
    reverse = direction in (ScanDirection.RIGHT_TO_LEFT, ScanDirection.BOTTOM_TO_TOP)
    height, width = image.shape[:2]
    partition_size, scan_size = (height, width) if horizontal else (width, height)
    if sample_count > partition_size:
        raise ValueError("sample_count must not exceed the partition dimension.")
    if scan_size < 2:
        raise ValueError("scan dimension must contain at least two pixels.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    count = int(sample_count)
    # 前几个分区接收余数，保证全覆盖、不重叠、无空条带，分区尺寸最多差1。
    base_size, remainder = divmod(partition_size, count)
    result = CaliperResult(config=config)
    start = 0
    for sample_index in range(count):
        stop = start + base_size + (1 if sample_index < remainder else 0)
        # 分区采用[start, stop)，其像素中心范围是start到stop-1。
        center = (start + stop - 1) / 2.0
        strip = gray[start:stop, :] if horizontal else gray[:, start:stop]
        # 与C++ cv::reduce的REDUCE_AVG一致，用浮点输出保留平均灰度精度。
        # dim=0平均多行，得到随X变化的剖面；dim=1平均多列，得到随Y变化的剖面。
        profile = cv2.reduce(strip, 0 if horizontal else 1, cv2.REDUCE_AVG,
                             dtype=cv2.CV_64F).reshape(-1)
        if reverse:
            profile = profile[::-1].copy()
        # 平滑仅沿扫描轴进行；sigma越大越能抑噪，也可能融合距离很近的边缘。
        if smooth_sigma > 0:
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 1), float(smooth_sigma),
                                       borderType=cv2.BORDER_REPLICATE).reshape(-1)
        # 配对前保留弱候选，不能因阈值过滤而跨过中间边缘组成虚假的宽条带。
        candidates = _profile_edges(profile, horizontal, reverse, center)
        edges, pairs = [], []
        if edge_mode is EdgeMode.SINGLE:
            edges = [edge for edge in candidates if edge.strength >= min_contrast
                     and (config.edge_polarity is EdgePolarity.BOTH or edge.polarity is config.edge_polarity)]
        else:
            # 候选已按扫描顺序排列，只配相邻异极性峰；both模式可共享中间边缘。
            # 例如暗→亮→暗→亮，前两边形成亮条，后两边形成暗条。
            for first, second in zip(candidates, candidates[1:]):
                if first.polarity is second.polarity:
                    continue
                pair_polarity = (PairPolarity.BRIGHT_STRIP if first.polarity is EdgePolarity.DARK_TO_BRIGHT
                                 else PairPolarity.DARK_STRIP)
                pair_width = second.position - first.position
                # 取较弱侧作为整对强度，避免一侧很强而另一侧不可靠；宽度上下限均含端点。
                strength = min(first.strength, second.strength)
                if (strength < min_contrast
                        or (config.pair_polarity is not PairPolarity.BOTH and config.pair_polarity is not pair_polarity)
                        or pair_width < min_width or (max_width is not None and pair_width > max_width)):
                    continue
                pairs.append(EdgePair(first=first, second=second, polarity=pair_polarity,
                                      strength=strength, width=pair_width,
                                      center=Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)))
        selected = edges if edge_mode is EdgeMode.SINGLE else pairs
        if selected and selection is not EdgeSelection.ALL:
            # 强度并列时保留扫描方向上最先遇到者，确保结果可重复。
            chosen = max(selected, key=lambda item: item.strength) if selection is EdgeSelection.STRONGEST else (
                selected[0] if selection is EdgeSelection.FIRST else selected[-1])
            selected = [chosen]
        # 即使没有候选也保留该分区；edges与pairs仅当前模式对应的列表可能非空。
        sample = CaliperSample(sample_index=sample_index, found=bool(selected),
                               strip_start=start, strip_stop=stop, strip_center=center,
                               edges=selected if edge_mode is EdgeMode.SINGLE else [],
                               pairs=selected if edge_mode is EdgeMode.PAIR else [])
        result.samples.append(sample)
        start = stop
    return result
