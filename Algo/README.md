# Algo 通用视觉算子库

`Algo` 是独立于当前产品工程的可复制 Python 算子库，不包含具体模型名、产品槽位、配方参数或 OK/NG 规则。

## 源码注释阅读指引（2026-10-13）

- `caliper/types.py`、`yolo/types.py`：枚举成员和结构体字段旁说明含义、单位、默认行为及适用模式，便于在PyCharm中查看参数。
- `caliper/edge.py`：公开接口说明输入/返回/异常，计算步骤旁解释分区与扫描轴、平均降维、梯度平台峰、反向坐标、边缘配对和选择规则。
- `yolo/model.py`：说明加载状态与实际推理成功的区别、单图/批量输入、后端颜色/尺寸约定、共享耗时及整批错误处理。
- `yolo/preprocess.py`、`postprocess.py`：说明实际尺寸取整、奇数补边、坐标逆变换、框与轮廓对应关系、贪心交集框合并及其限制。
- 本轮仅补充中文注释和文档字符串；去除文档字符串后的语法树比较确认执行代码不变。

## 目录

```text
Algo/
  yolo/
    model.py        # 模型上下文、加载模型、单图和批量推理
    preprocess.py   # 仅内存图像缩放、补边、通道转换
    postprocess.py  # 结果解析、坐标还原、交集框合并
    types.py        # 枚举、配置、变换记录和结果结构
  caliper/
    edge.py         # 四方向条带平均找边，支持单边及边缘对
    types.py        # 枚举、配置和结果结构体
  tests/
    test_caliper.py # 卡尺合成图回归验证
    test_yolo.py    # YOLO预后处理、结构和后端接口验证
```

## YOLO 使用方式

YOLO只接收调用方已经准备好的内存图像，不提供读图函数或图片路径参数。模型文件路径仍由 `load_model` 接收，加载、模型上下文、单图与批量推理全部位于 `model.py`。

### PyCharm检测和分割例程

- 目标检测：[examples/yolo_detection_example.py](examples/yolo_detection_example.py)，显示检测框、类别和置信度。
- 实例分割：[examples/yolo_segmentation_example.py](examples/yolo_segmentation_example.py)，显示半透明区域、轮廓、框、类别和置信度。

选择 `gpu-1060` 解释器，修改各文件顶部 `MODEL_PATH`、`IMAGE_PATH` 后直接运行。`MODEL_CONFIG` 配置任务/设备，`INFERENCE_CONFIG` 配置输入尺寸和阈值，固定选项通过枚举补全。示例默认CPU、640输入、置信度0.25；模型路径为占位路径，需要替换为匹配任务的实际模型。

读图只发生在例程中，`infer`接收内存数组。运行后保存PNG与JSON到原图旁的 `yolo_detection_results` 或 `yolo_segmentation_results`，重复运行自动加序号；可通过 `OUTPUT_DIR` 指定目录。`SHOW_WINDOW=True` 显示左侧原图、右侧结果，按Esc/Q或关闭窗口退出；False只保存文件。

显示保存逻辑由两个例程共用 `yolo_example_output.py`，不属于算法库核心。JSON保存完整小数坐标与分割轮廓；图中填充仅为可视化。OpenCV内置字体不能绘制中文类别时，以class_id显示，JSON保留完整类别名。零目标也正常保存和显示。

两个例程已通过本地检测/分割ONNX的CPU空白图端到端验证；合成正目标验证框、轮廓填充和JSON输出，预览使用GUI替身检查，未人工验证实际桌面弹窗。

```python
from Algo.yolo import (
    YoloTask, YoloDevice, YoloStatus, PreprocessMode,
    YoloModelConfig, YoloInferenceConfig, PreprocessConfig, load_model, infer, infer_batch,
)

# 模型配置使用枚举；加载本地文件，不自动下载模型。
model = load_model(
    r"D:\models\part.onnx",
    YoloModelConfig(task=YoloTask.DETECT, device=YoloDevice.CPU),
)
if model.status is not YoloStatus.OK:
    raise RuntimeError(model.message)

# image是上游算子输出的uint8灰度图或BGR三通道数组。
config = YoloInferenceConfig(
    preprocess=PreprocessConfig(mode=PreprocessMode.RESIZE_PADDING, target_size=640),
    confidence_threshold=0.25,
    iou_threshold=0.45,
)
result = infer(model, image, config)
if result.status is not YoloStatus.OK:
    raise RuntimeError(result.message)
for detection in result.detections:
    print(detection.label, detection.confidence, detection.box.x, detection.box.width)

# 批量接收图像列表/元组，每张返回一个独立结果结构。
results = infer_batch(model, [image_a, image_b], config)
data = result.to_dict()  # 仅保存JSON时转换枚举和嵌套结构。
```

### 模块边界

| 模块 | 公开能力 |
|---|---|
| `model.py` | `YoloModel`、`load_model`、`infer`、`infer_batch` |
| `preprocess.py` | `prepare_image(image, PreprocessConfig)`，返回处理图及 `ImageTransform` |
| `postprocess.py` | `parse_result`、`restore_box/restore_boxes/restore_polygon`、`merge_same_label_intersections` |
| `types.py` | 模型/设备/预处理/状态枚举，以及配置、框、轮廓和结果结构 |

所有接口仍可从 `Algo.yolo` 导入。预处理只负责图像转换和记录变换，坐标逆变换只属于后处理。`infer`及`infer_batch`内部自动调用预后处理，普通调用方不必手工串联。

### 参数与输入约定

- `YoloModelConfig`：`task=YoloTask.DETECT`（或SEGMENT）；`device=YoloDevice.AUTO`（或CPU/CUDA）；`gpu_index=0`。AUTO传递为后端默认设备，CUDA将编号交给后端，不内置GPU失败自动回退。
- `PreprocessConfig`：`mode=PreprocessMode.RESIZE_PADDING`、`target_size=640`、`padding_color=(114,114,114)`、`convert_bgr_to_rgb=False`。
- `RESIZE_PADDING`等比例缩放后补成正方形；`PADDING`仅补边，输入大于目标尺寸时报错；`NONE`跳过本库的缩放补边。`target_size`仍作为后端 `imgsz` 请求，后端可能按模型固定尺寸/步长调整，也会执行必要的张量预处理。
- 输入只能是非空uint8灰度或BGR三通道NumPy数组；灰度复制为三通道，不修改调用方图像。RGB图应由调用方先转成BGR。
- `convert_bgr_to_rgb=True`时，独立 `prepare_image` 返回RGB；模型适配层会将其恢复为Ultralytics NumPy入口要求的BGR，再由后端统一转张量，避免重复换色。该参数不是输入RGB声明，也不用于颠倒模型颜色。
- `YoloInferenceConfig`：嵌套 `preprocess`、`confidence_threshold=0.25`、`iou_threshold=0.45`、`merge_same_label=False`、`include_segments=True`。单图和批量使用相同配置和合并规则。
- 配置是只读dataclass，固定选项必须使用枚举。类型/非法输入抛出TypeError或ValueError；模型文件缺失、加载失败或后端执行失败通过 `YoloStatus.ERROR` 和 `message` 返回。
- `infer_batch`接受图像列表或元组，不接受路径、生成器或直接传一张NumPy数组。空列表返回空列表；真实批量能力受模型导出batch维度约束，批次个数错误或后端错误返回逐图错误结果。

### 坐标及结果

**“输入图坐标”仅指本次传入YOLO的图像。** 如果上游传入裁剪ROI，结果坐标就在ROI内，不会自动加回相机整图偏移。预处理记录实际取整后的横纵尺寸，还原分别使用 `scale_x/scale_y`，避免非整数缩放引起坐标漂移。

- `YoloBox(x,y,width,height)`：唯一公开矩形格式，单位px；不再隐式猜测xyxy字典或四元组。
- `ImageTransform`：原尺寸、实际缩放尺寸、四侧补边，提供 `scale_x/scale_y` 属性。
- `restore_box/restore_boxes`接收 `YoloBox`；`restore_polygon`接受Nx2数组、坐标序列或 `YoloPoint` 列表。默认裁剪到输入图边界，`clip=False`可保留越界坐标。
- `YoloResult`：`status/message/model_path/config/detections/segments/raw_detections/transform/timing`，无图片路径字段。
- `YoloDetection`：`label/class_id/confidence/box/preprocessed_box/merged_count`。
- `YoloSegment`：`label/class_id/confidence/box/polygon`，轮廓为 `YoloPoint` 列表，不输出像素Mask。
- `result.to_dict()`转换为独立JSON兼容字典；`config.to_dict()`也可单独保存实际配置。

同类框合并保留历史**交集框**规则，按置信度降序处理；它不是NMS或并集。开启时 `raw_detections`保留原始检测，分割轮廓及对应框始终来自原始检测，不能按合并后的检测列表重新逐项配对。

`timing.preprocess_ms/inference_ms`在批量调用中是整批共享耗时，`batch_size`记录批次数量，不能按返回图片数量累加。`postprocess_ms`为当前图片的后处理耗时。推理耗时统计完整 `predict()`，包括后端内部处理，不是纯ONNX `session.run()` 时间。

### 迁移与验证

2026-10-13：删除 `inference.py`，模型相关功能集中到 `model.py`；删除 `load_image`、`image_path`；`restore_*`移到后处理。旧字符串及散列参数调用迁移为枚举+配置，原字典索引改成属性访问。未修改工程 `WindowsApplication1/python_algorithms/operators` 的模型算子和在线调用。

`Algo/tests/test_yolo.py`共26项测试覆盖预处理模式、实际比例还原、NumPy轮廓、后端输入颜色与尺寸、单图/批量一致性、错误状态、配置校验及文件职责。与卡尺合计88项测试通过。另在 `gpu-1060` 环境以CPU运行本地检测/分割ONNX模型的内存空白图验证，均成功；这不代表真实图片检测精度或GPU/所有导出格式均已验证。

## 卡尺使用方式

公开入口为 `caliper_find_edges(image, config=None) -> CaliperResult`。固定选项使用枚举，配置使用只读 `CaliperConfig`，结果使用结构体；不再接收字符串选项或旧版散列关键字参数。省略配置时采用 `CaliperConfig()` 默认值。

```python
from Algo.caliper import (
    CaliperConfig, ScanDirection, EdgeMode, EdgeSelection,
    EdgePolarity, PairPolarity, caliper_find_edges,
)

# image由调用方提供，是uint8灰度图或BGR图；整张输入图即检测区域。
config = CaliperConfig(
    direction=ScanDirection.LEFT_TO_RIGHT,
    edge_mode=EdgeMode.PAIR,
    pair_polarity=PairPolarity.BRIGHT_STRIP,
    selection=EdgeSelection.STRONGEST,
    sample_count=4,
    min_contrast=8.0,
    min_width=2.0,
    max_width=60.0,
)
result = caliper_find_edges(image, config)

# 嵌套结果使用属性访问，PyCharm可自动提示字段。
for sample in result.samples:
    for pair in sample.pairs:
        print(pair.first.x, pair.second.x, pair.center.x, pair.width)

# 仅在存储或传输边界转换字典，枚举自动转换为稳定英文值。
data = result.to_dict()
```

### PyCharm图片示例

打开 [examples/caliper_example.py](examples/caliper_example.py)，选择 `gpu-1060` 解释器，修改顶部 `IMAGE_PATH` 与 `CONFIG = CaliperConfig(...)` 后直接运行，无需命令行参数或指定工作目录。示例保留用户已设置的图片路径和参数；库默认参数以本页表格为准。

- 输入 `ScanDirection.`、`EdgeSelection.` 等即可用IDE补全选择固定成员。
- `OUTPUT_DIR=None` 表示保存到原图旁的 `caliper_results`；标记PNG及JSON重复运行加序号，不覆盖原图和已有结果。
- `SHOW_WINDOW=True` 时保存后弹窗：左侧原图、右侧结果。按Esc/Q或关闭窗口结束；False时只保存。
- 预览大图自动缩小，保存图仍保持原始分辨率；显示需要支持GUI的OpenCV。
- 标记图的灰线是分区边界；绿色十字表示单边或边缘对第一侧，橙色是第二侧，紫色是中心并标注宽度。不会拟合或连接不同分区的边缘。
- 示例JSON包含 `source_image`、`parameters=config.to_dict()` 及 `result=result.to_dict()`。

### 固定选项

| 枚举 | 成员 |
|---|---|
| `ScanDirection` | `LEFT_TO_RIGHT`、`RIGHT_TO_LEFT`、`TOP_TO_BOTTOM`、`BOTTOM_TO_TOP` |
| `EdgeMode` | `SINGLE`、`PAIR` |
| `EdgeSelection` | `ALL`、`STRONGEST`、`FIRST`、`LAST` |
| `EdgePolarity` | `BOTH`、`BRIGHT_TO_DARK`、`DARK_TO_BRIGHT` |
| `PairPolarity` | `BOTH`、`BRIGHT_STRIP`、`DARK_STRIP` |

枚举值用于JSON存储，例如 `ScanDirection.LEFT_TO_RIGHT.value` 为 `"left_to_right"`。Python API要求传枚举，不自动接受字符串。若需要界面下拉框，可用 `list(ScanDirection)` 获取选项；从外部配置读取时调用 `ScanDirection(stored_value)` 显式转换，未知值会报错。

### CaliperConfig参数

| 字段 | 默认值 | 含义 |
|---|---|---|
| `direction` | `ScanDirection.LEFT_TO_RIGHT` | 扫描方向 |
| `edge_mode` | `EdgeMode.SINGLE` | 查找单边或边缘对 |
| `selection` | `EdgeSelection.ALL` | 每个条带独立选择全部、最强、首个、末个 |
| `edge_polarity` | `EdgePolarity.BOTH` | 仅单边模式使用 |
| `pair_polarity` | `PairPolarity.BOTH` | 仅边缘对模式使用 |
| `sample_count` | `1` | 正整数；左右扫描不超过图像高度，上下扫描不超过图像宽度 |
| `min_contrast` | `8.0` | 平均、平滑后相邻灰度差的绝对值阈值，包含端点 |
| `smooth_sigma` | `0.0` | 一维高斯平滑标准差，单位px；0关闭 |
| `min_width` | `0.0` | 边缘对最小宽度，单位px，包含端点 |
| `max_width` | `None` | 边缘对最大宽度，单位px，包含端点；None不设上限 |

构造时检查枚举类型、正整数采样数量、有限非负数值及宽度上下限。模式切换时只读取对应极性字段，未使用的字段也必须类型合法。宽度参数仅在边缘对模式筛选。配置冻结以避免创建后绕过校验；需要修改时重建，或用 `dataclasses.replace(config, sample_count=8)`，同样会重新校验。

图像为非空 `uint8` 二维灰度或三通道BGR数组，扫描方向至少2像素；执行时检查采样数量是否超过图像维度。类型错误抛 `TypeError`，数值/图像约束错误抛 `ValueError`。合法输入未找到目标正常返回 `result.found=False`。

### 采样与找边规则

- 左右扫描按高度平分为水平条带，对各行平均；上下扫描按宽度平分为竖直条带，对各列平均。使用 `cv2.reduce(..., REDUCE_AVG, dtype=CV_64F)`。
- 每个像素恰好属于一个条带。不能整除时前几个条带多1像素，编号始终按上到下或左到右排列；反向扫描仅改变条带内搜索顺序。
- `sample_count=1` 对整张图降维；增大数量保留更多局部变化，降噪能力降低。宽条带可能将倾斜/弯曲边缘平均模糊。
- 可选平滑后用相邻差分提取梯度幅值局部峰。连续同极性等高平台合为一个峰，位置取平台中心；不把所有超阈值像素各自输出。
- 极性和FIRST/LAST相对于扫描方向定义；同一左亮右暗边界，向右为亮到暗，向左为暗到亮。
- 配对使用未做强度过滤的相邻候选峰，两峰需极性相反、强度均达标。不跳过弱候选拼接更宽边缘对；噪声较多时可通过平滑抑制。
- 亮条按扫描顺序为暗→亮→暗，暗条为亮→暗→亮。BOTH+ALL允许相邻亮暗条共享边界；不输出只有一侧可见的不完整边缘对。
- 边缘对强度取两侧较小值，宽度约束先于选择执行。STRONGEST并列时保留扫描方向最先遇到者。

### 结果结构

| 结构 | 字段/属性 |
|---|---|
| `CaliperResult` | `config`、`samples`、`found`、`sample_count`、`found_sample_count`、`to_dict()` |
| `CaliperSample` | `sample_index`、`found`、`strip_start`、`strip_stop`、`strip_center`、`edges: List[Edge]`、`pairs: List[EdgePair]` |
| `Edge` | `x`、`y`、`position`、`strength`、`polarity: EdgePolarity` |
| `EdgePair` | `first: Edge`、`second: Edge`、`center: Point`、`width`、`strength`、`polarity: PairPolarity` |
| `Point` | `x`、`y` |

`edges/pairs`始终存在，非当前模式的列表为空。`found_sample_count / sample_count` 可计算命中比例，但不代表产品合格。结果保留本次配置快照；配置、点、边缘和边缘对为只读数据。

分区为左闭右开 `[strip_start, strip_stop)`，中心取 `(start + stop - 1) / 2`。`x/y`相对输入图像左上角，未扫描维度取条带中心；`position`从扫描起点计算，反向时不同于原图轴坐标。小数来自差分位置或平台中心，不代表亚像素模型拟合；宽度单位px。

`to_dict()` 生成独立JSON兼容快照，嵌套枚举转换为英文值。返回字典包含 `config`、`samples`、`status="Ok"`、`found`、`sample_count`、`found_sample_count`。状态仅表示函数正常返回，不表示产品合格；结果对象不需要检查字符串状态，执行失败会抛异常。

### 迁移与验证

2026-10-13：统一使用枚举和结构体。原 `caliper_find_edges(image, direction="...", ...)` 改为 `caliper_find_edges(image, CaliperConfig(...))`；原统一 `polarity` 拆为 `edge_polarity/pair_polarity`。原字典索引改为属性访问，保存时调用 `to_dict()`；方向、模式等参数移至结果的 `config` 中。

原 `find_edges` 名称及 `measure_edge_line/measure_line_center/measure_paired_edges` 接口不再导出。不需要参考线、不拟合直线、不提供产品 `is_ok`；工程 `operators/caliper.py` 及业务调用不变。

```powershell
# 在仓库根目录运行独立库卡尺验证。
& D:/anaconda3/envs/recon/python.exe -m pytest Algo/tests/test_caliper.py -q -p no:cacheprovider
```

62项测试覆盖原四方向/分区/强度/配对行为，以及枚举校验、配置不可变性、结构体访问、JSON快照及PyCharm示例的单边/边缘对图片输出。示例预览通过GUI替身验证，未实际打开桌面窗口。合成图不替代真实项目样图调参。

## 依赖

- NumPy
- OpenCV-Python
- Ultralytics（仅在调用 `load_model` 时需要）
- ONNX 模型还需要 `onnx` 和 `onnxruntime`；库不会自动安装缺失依赖

当前未修改 `WindowsApplication1/python_algorithms` 下的旧算子，`Algo` 可单独复制到后续项目使用。
