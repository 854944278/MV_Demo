# -*- coding: utf-8 -*-
"""卡尺的固定选项、配置和测量结果；运行时使用类型，存储时转换为普通字典。"""

import math
import numbers
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional


class ScanDirection(Enum):
    """灰度扫描方向，决定首末顺序和边缘极性。"""
    LEFT_TO_RIGHT = "left_to_right"  # 沿X递增扫描，常用于找竖向边缘。
    RIGHT_TO_LEFT = "right_to_left"  # 沿X递减扫描；同一物理边界的极性随之反转。
    TOP_TO_BOTTOM = "top_to_bottom"  # 沿Y递增扫描，常用于找横向边缘。
    BOTTOM_TO_TOP = "bottom_to_top"  # 沿Y递减扫描，首个边缘指从下侧最先遇到者。


class EdgeMode(Enum):
    """查找单侧边缘或完整边缘对。"""
    SINGLE = "single"  # 每条剖面输出独立边缘。
    PAIR = "pair"  # 相邻反极性边缘配对，输出宽度与中心。


class EdgeSelection(Enum):
    """每个采样条带内的目标选择方式。"""
    ALL = "all"  # 保留全部有效目标，仍按扫描顺序排列。
    STRONGEST = "strongest"  # 选强度最大者，并列取最先遇到者。
    FIRST = "first"  # 通过极性/强度/宽度筛选后的首个目标。
    LAST = "last"  # 通过筛选后的最后一个目标。


class EdgePolarity(Enum):
    """单边缘沿扫描方向的灰度变化。"""
    BOTH = "both"  # 不限制单边极性。
    BRIGHT_TO_DARK = "bright_to_dark"  # 沿扫描方向灰度下降。
    DARK_TO_BRIGHT = "dark_to_bright"  # 沿扫描方向灰度上升。


class PairPolarity(Enum):
    """边缘对包围的亮条或暗条类型。"""
    BOTH = "both"  # 亮条和暗条均可，相邻条带目标可共享边界。
    BRIGHT_STRIP = "bright_strip"  # 暗→亮→暗，两侧包围亮区域。
    DARK_STRIP = "dark_strip"  # 亮→暗→亮，两侧包围暗区域。


def _to_plain(value):
    """递归转换asdict后的枚举，供配置和结果共用，输出可直接保存为JSON。"""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class CaliperConfig:
    """经过校验的只读参数结构；修改时重新构造或使用dataclasses.replace。"""
    direction: ScanDirection = ScanDirection.LEFT_TO_RIGHT  # 扫描方向，不是被测边缘的延伸方向。
    edge_mode: EdgeMode = EdgeMode.SINGLE  # 控制输出edges还是pairs。
    selection: EdgeSelection = EdgeSelection.ALL  # 每个条带分别选择，不在整张图所有条带间竞争。
    edge_polarity: EdgePolarity = EdgePolarity.BOTH  # 只在SINGLE模式筛选，但两种模式均校验枚举类型。
    pair_polarity: PairPolarity = PairPolarity.BOTH  # 只在PAIR模式筛选亮条/暗条。
    sample_count: int = 1  # 正整数；垂直扫描轴的分区数，1表示整图平均。
    min_contrast: float = 8.0  # 相邻平均灰度差的最低绝对值，含阈值端点，单位为灰度级。
    smooth_sigma: float = 0.0  # 扫描轴高斯平滑标准差(px)，0关闭；增大会降低尖锐响应。
    min_width: float = 0.0  # 边缘对间距下限(px)，含端点，单边模式不参与筛选。
    max_width: Optional[float] = None  # 边缘对间距上限(px)，含端点；None不限制。

    def __post_init__(self):
        # 类型标注本身不做运行时校验，主动拒绝字符串和其他枚举类型。
        for name, enum_type in (("direction", ScanDirection), ("edge_mode", EdgeMode),
                                ("selection", EdgeSelection), ("edge_polarity", EdgePolarity),
                                ("pair_polarity", PairPolarity)):
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(name + " must be " + enum_type.__name__ + ".")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, numbers.Integral):
            raise TypeError("sample_count must be an integer.")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive.")
        object.__setattr__(self, "sample_count", int(self.sample_count))
        # 数值归一化为Python标量，避免NumPy参数进入JSON时不可序列化。
        for name in ("min_contrast", "smooth_sigma", "min_width", "max_width"):
            value = getattr(self, name)
            if name == "max_width" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(name + " must be a real number.")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(name + " must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        if self.max_width is not None and self.max_width < self.min_width:
            raise ValueError("max_width must be greater than or equal to min_width.")

    def to_dict(self) -> dict:
        """转换存储格式；算法调用仍要求传入本结构体。"""
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class Point:
    """输入图像坐标系中的像素位置。"""
    x: float  # 输入图横坐标(px)，原点在左上角。
    y: float  # 输入图纵坐标(px)，向下递增。


@dataclass(frozen=True)
class Edge:
    """单个边缘的位置、强度及实际极性。"""
    x: float  # 输入图横坐标(px)，非扫描轴位置取条带中心。
    y: float  # 输入图纵坐标(px)，不是条带编号。
    position: float  # 距扫描起点的剖面坐标(px)，反向时不等于原图轴坐标。
    strength: float  # 局部梯度峰的绝对值；不是置信度或区域面积。
    polarity: EdgePolarity  # 实际检测到的极性，结果中不会是BOTH。


@dataclass(frozen=True)
class EdgePair:
    """按扫描顺序排列的两侧边缘及中心、宽度。"""
    first: Edge  # 按扫描顺序的第一侧，反向时可能是物理右/下侧。
    second: Edge  # 按扫描顺序的第二侧。
    center: Point  # 两边缘坐标中点，不是拟合线或区域重心。
    width: float  # second.position-first.position，单位px。
    strength: float  # 两侧强度的较小值，要求两侧都可靠。
    polarity: PairPolarity  # 实际亮条/暗条类型，结果中不会是BOTH。


@dataclass
class CaliperSample:
    """一个采样条带的边界及测量结果；各实例独立持有结果列表。"""
    sample_index: int  # 条带编号，从0开始，反向扫描不改变编号顺序。
    strip_start: int  # 分区起始行/列，包含此位置。
    strip_stop: int  # 分区结束行/列，不包含此位置。
    strip_center: float  # 另一轴像素中心坐标：(start+stop-1)/2。
    found: bool  # 本条带至少有一个最终选中的边缘或边缘对。
    edges: List[Edge] = field(default_factory=list)  # 单边结果；PAIR模式为空，不与pairs重复输出。
    pairs: List[EdgePair] = field(default_factory=list)  # 边缘对结果；SINGLE模式为空。


@dataclass
class CaliperResult:
    """完整测量结果；配置记录本次参数，found不代表产品合格。"""
    config: CaliperConfig  # 本次使用的只读配置，可与其他调用安全共享。
    samples: List[CaliperSample] = field(default_factory=list)  # 所有条带的结果，包括found=False的条带。

    @property
    def sample_count(self) -> int:
        """返回实际条带数量。"""
        return len(self.samples)

    @property
    def found_sample_count(self) -> int:
        """由采样结果计算命中数量，避免重复维护状态。"""
        return sum(sample.found for sample in self.samples)

    @property
    def found(self) -> bool:
        """至少一个条带找到目标时为True。"""
        return any(sample.found for sample in self.samples)

    def to_dict(self) -> dict:
        """生成独立的JSON兼容快照，序列化不会改变原始枚举和数据。"""
        data = _to_plain(asdict(self))
        data.update(status="Ok", found=self.found, sample_count=self.sample_count,
                    found_sample_count=self.found_sample_count)
        return data
