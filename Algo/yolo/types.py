# -*- coding: utf-8 -*-
"""YOLO配置枚举、坐标变换及输出结构；与模型后端解耦。"""

import math
import numbers
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class YoloTask(Enum):
    """本库已封装的任务类型。"""
    DETECT = "detect"  # 目标检测：输出类别、置信度和轴对齐矩形框。
    SEGMENT = "segment"  # 实例分割：额外解析每个实例的多边形轮廓。


class YoloDevice(Enum):
    """推理设备策略，AUTO交给后端选择。"""
    AUTO = "auto"  # 交由后端选择设备，本库不额外探测。
    CPU = "cpu"  # 请求CPU执行。
    CUDA = "cuda"  # 请求指定gpu_index的CUDA设备，不自动回退CPU。


class PreprocessMode(Enum):
    """仅控制本库的图像整理，NONE仍允许后端执行其必要预处理。"""
    RESIZE_PADDING = "resize_padding"  # 等比例缩放长边到target_size，再居中补成正方形。
    PADDING = "padding"  # 仅补边；输入任一尺寸超过target_size时报错。
    NONE = "none"  # 本库不缩放/补边，但后端仍会做自身必要的预处理。


class YoloStatus(Enum):
    """执行状态，不代表产品OK/NG。"""
    OK = "Ok"  # 执行成功，检测数量为0也属于OK。
    ERROR = "ErrorState"  # 模型加载或推理失败，原因见message，不是产品NG。


def _plain(value):
    """配置和结果共用的JSON转换，仅在存储边界将枚举变成字符串。"""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class YoloModelConfig:
    """模型任务和设备配置，GPU编号仅在CUDA模式使用。"""
    task: YoloTask = YoloTask.DETECT  # 必须与模型实际任务匹配，不能用切换枚举改变模型本身。
    device: YoloDevice = YoloDevice.AUTO  # 设备选择策略；具体能否执行由后端和安装环境决定。
    gpu_index: int = 0  # 非负设备编号，仅CUDA模式使用。

    def __post_init__(self):
        # 固定选项不接受字符串，避免无效参数在推理时才暴露。
        if not isinstance(self.task, YoloTask) or not isinstance(self.device, YoloDevice):
            raise TypeError("task/device must use YoloTask/YoloDevice.")
        if isinstance(self.gpu_index, bool) or not isinstance(self.gpu_index, numbers.Integral):
            raise TypeError("gpu_index must be an integer.")
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be non-negative.")
        object.__setattr__(self, "gpu_index", int(self.gpu_index))

    def to_dict(self) -> dict:
        """保存可序列化配置。"""
        return _plain(asdict(self))


@dataclass(frozen=True)
class PreprocessConfig:
    """图像缩放补边配置，target_size也作为后端的推理尺寸请求。"""
    mode: PreprocessMode = PreprocessMode.RESIZE_PADDING  # 本库的几何预处理策略。
    target_size: int = 640  # 正整数像素边长，同时传给后端imgsz；实际尺寸可能受模型约束。
    padding_color: Tuple[int, int, int] = (114, 114, 114)  # 8位BGR三元组，先补边再按开关转换通道。
    convert_bgr_to_rgb: bool = False  # 控制prepare_image输出顺序；不是声明输入为RGB。

    def __post_init__(self):
        if not isinstance(self.mode, PreprocessMode):
            raise TypeError("mode must be PreprocessMode.")
        if isinstance(self.target_size, bool) or not isinstance(self.target_size, numbers.Integral):
            raise TypeError("target_size must be an integer.")
        if self.target_size < 1:
            raise ValueError("target_size must be positive.")
        object.__setattr__(self, "target_size", int(self.target_size))
        # 补边统一用BGR三元组；灰度输入会在预处理转换成三通道。
        if not isinstance(self.padding_color, tuple) or len(self.padding_color) != 3:
            raise TypeError("padding_color must be a three-element tuple.")
        for value in self.padding_color:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= 255:
                raise ValueError("padding_color values must be integers in [0, 255].")
        object.__setattr__(self, "padding_color", tuple(int(v) for v in self.padding_color))
        if not isinstance(self.convert_bgr_to_rgb, bool):
            raise TypeError("convert_bgr_to_rgb must be bool.")


@dataclass(frozen=True)
class YoloInferenceConfig:
    """单图和批量共用推理配置，阈值与模型加载解耦。"""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)  # 本次图像预处理配置，默认每个实例独立创建。
    confidence_threshold: float = 0.25  # 传给后端的候选置信度阈值，范围[0,1]。
    iou_threshold: float = 0.45  # 传给后端的IoU阈值，具体作用取决于模型/后端，不控制交集框合并。
    merge_same_label: bool = False  # 可选历史交集框合并；不合并分割轮廓。
    include_segments: bool = True  # 是否解析分割轮廓，不会把检测模型变成分割模型。

    def __post_init__(self):
        if not isinstance(self.preprocess, PreprocessConfig):
            raise TypeError("preprocess must be PreprocessConfig.")
        for name in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(name + " must be a number.")
            if not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(name + " must be in [0, 1].")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.merge_same_label, bool) or not isinstance(self.include_segments, bool):
            raise TypeError("merge_same_label/include_segments must be bool.")

    def to_dict(self) -> dict:
        """保存包括嵌套预处理在内的实际参数。"""
        return _plain(asdict(self))


@dataclass(frozen=True)
class ImageTransform:
    """从本次输入图到预处理图的变换，不含上游ROI偏移。"""
    original_width: int  # 本次输入图宽度，不含上游ROI整图偏移。
    original_height: int  # 本次输入图高度。
    resized_width: int  # 实际取整后的缩放宽度，用于精确计算scale_x。
    resized_height: int  # 实际取整后的缩放高度，用于精确计算scale_y。
    pad_left: int = 0  # 补在图像左侧的像素数，逆变换时先减去。
    pad_top: int = 0  # 补在图像上侧的像素数，逆变换时先减去。
    pad_right: int = 0  # 右侧补边；奇数总补边时可能比左侧多1。
    pad_bottom: int = 0  # 下侧补边；奇数总补边时可能比上侧多1。

    def __post_init__(self):
        # 后处理公开函数也可接收调用方构造的变换，因此禁止零尺寸或负补边。
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(name + " must be an integer.")
            if value < (0 if name.startswith("pad_") else 1):
                raise ValueError("invalid transform dimension: " + name)
            object.__setattr__(self, name, int(value))

    @property
    def scale_x(self) -> float:
        """按实际取整后的尺寸计算横向比例。"""
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        """纵向独立记录，避免缩放取整导致还原偏移。"""
        return self.resized_height / self.original_height


@dataclass(frozen=True)
class YoloPoint:
    """输入图像内的轮廓点坐标。"""
    x: float  # 本次输入图横坐标(px)，左上为原点。
    y: float  # 本次输入图纵坐标(px)，向下递增。


@dataclass(frozen=True)
class YoloBox:
    """统一xywh矩形，单位为像素。"""
    x: float  # 矩形左上角横坐标(px)。
    y: float  # 矩形左上角纵坐标(px)。
    width: float  # 几何宽度(px)，不是右下角横坐标。
    height: float  # 几何高度(px)，不是右下角纵坐标。


@dataclass(frozen=True)
class YoloDetection:
    """检测框及模型类别信息，合并不修改原始结果。"""
    label: str  # 模型类别名称；缺失名称时回退为类别编号文本。
    class_id: int  # 模型输出的类别编号。
    confidence: float  # 模型输出的置信度。
    box: YoloBox  # 还原至本次输入图的xywh框。
    preprocessed_box: Optional[YoloBox] = None  # 合并前的后端坐标框，供变换排查；合并后不重新计算。
    merged_count: int = 1  # 当前交集框累计合并的原始框数量，未合并为1。


@dataclass
class YoloSegment:
    """实例分割轮廓；保持与原始检测框对应，不参与框合并。"""
    label: str  # 对应原始检测实例的类别名。
    class_id: int  # 对应原始检测实例的类别编号。
    confidence: float  # 对应原始检测实例的置信度。
    box: YoloBox  # 对应原始检测框，不跟随合并后的detections列表改变。
    polygon: List[YoloPoint] = field(default_factory=list)  # 输入图坐标下的轮廓点列表，不是像素Mask。


@dataclass
class YoloTiming:
    """毫秒耗时；批量预处理和推理属于整批，不得逐图累加。"""
    preprocess_ms: float = 0.0  # 本库预处理耗时(ms)，批量时为整批共享值。
    inference_ms: float = 0.0  # 完整predict耗时(ms)，包含后端内部处理，非纯网络耗时。
    postprocess_ms: float = 0.0  # 仅当前图片结果解析与可选合并耗时(ms)。
    batch_size: int = 1  # 本次请求的图像数量，提示共享耗时不能逐图累加。


@dataclass
class YoloResult:
    """单张结果结构，坐标均相对于本次传入YOLO的图像。"""
    status: YoloStatus  # 先检查执行状态，再消费检测结果；无目标与失败分开表示。
    message: str  # 执行说明或失败原因。
    model_path: str  # 模型文件路径，不是图片路径。
    config: YoloInferenceConfig  # 本次推理配置，记录阈值和预处理参数。
    detections: List[YoloDetection] = field(default_factory=list)  # 最终检测框，开启合并时为交集框结果。
    segments: List[YoloSegment] = field(default_factory=list)  # 原始实例分割轮廓；关闭解析或检测任务时通常为空。
    raw_detections: List[YoloDetection] = field(default_factory=list)  # 仅开启框合并时保存原检测，否则为空。
    transform: Optional[ImageTransform] = None  # 本次输入到预处理图的变换，后处理已据此还原坐标。
    timing: YoloTiming = field(default_factory=YoloTiming)  # 耗时明细；错误结果当前保留默认值，不能用于性能统计。

    def to_dict(self) -> dict:
        """递归转换枚举与结构体，用于JSON或跨语言传输。"""
        return _plain(asdict(self))
