# -*- coding: utf-8 -*-
"""读取指定图片执行卡尺找边，输出标记图和JSON测量数据。"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

# 直接在PyCharm运行此文件时，按文件位置定位库，不依赖工作目录设置。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Algo.caliper import (CaliperConfig, ScanDirection, EdgeMode, EdgeSelection,
                          EdgePolarity, PairPolarity, caliper_find_edges)


# ==================== 在这里修改图片路径和参数 ====================
IMAGE_PATH = r"D:\00image\1.png"  # 替换成自己的8位灰度图或BGR图片。
OUTPUT_DIR = None  # None表示保存到原图旁的caliper_results目录。
SHOW_WINDOW = True  # 运行完成后显示原图与找边结果。

# 固定选项使用枚举，在PyCharm输入枚举名和点号即可选择。
CONFIG = CaliperConfig(
    direction=ScanDirection.LEFT_TO_RIGHT,
    edge_mode=EdgeMode.SINGLE,
    selection=EdgeSelection.STRONGEST,
    edge_polarity=EdgePolarity.DARK_TO_BRIGHT,  # 仅单边模式使用。
    pair_polarity=PairPolarity.BOTH,  # 仅边缘对模式使用。
    sample_count=10,  # 垂直于扫描方向均分的条带数量。
    min_contrast=8.0,  # 边缘最低灰度变化强度。
    smooth_sigma=0.0,  # 一维高斯平滑，0表示关闭。
    min_width=0.0,  # 边缘对最小宽度，单位px。
    max_width=None,  # None表示不限制边缘对最大宽度。
)
# ================================================================


def main():
    # 参数集中在文件顶部，PyCharm直接点击运行即可，无需配置命令行参数。
    # 字节解码兼容Windows中文路径；保持原位深，由算子校验输入而非静默转换。
    try:
        source_path = Path(IMAGE_PATH).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"请先修改文件顶部的IMAGE_PATH，图片不存在：{source_path}")
        encoded = np.fromfile(str(source_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
        result = caliper_find_edges(image, CONFIG)
    except (OSError, TypeError, ValueError, cv2.error) as exc:
        raise RuntimeError(f"卡尺示例执行失败：{exc}") from exc

    # 绘图仅在副本上操作；每个点位均来自实际测量，不连接不同分区、不拟合直线。
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    horizontal = CONFIG.direction in (ScanDirection.LEFT_TO_RIGHT, ScanDirection.RIGHT_TO_LEFT)
    for sample in result.samples:
        boundary = sample.strip_start
        if boundary > 0:
            first = (0, boundary) if horizontal else (boundary, 0)
            second = (image.shape[1] - 1, boundary) if horizontal else (boundary, image.shape[0] - 1)
            cv2.line(annotated, first, second, (120, 120, 120), 1)
        # 单边用绿色；边缘对第一侧为绿色、第二侧为橙色，中心为紫色。
        markers = [(edge, (0, 255, 0)) for edge in sample.edges]
        for pair in sample.pairs:
            markers.extend([(pair.first, (0, 255, 0)), (pair.second, (0, 165, 255)),
                            (pair.center, (255, 0, 255))])
            center = (int(round(pair.center.x)), int(round(pair.center.y)))
            cv2.putText(annotated, f"w={pair.width:.2f}px", (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        for point, color in markers:
            position = (int(round(point.x)), int(round(point.y)))
            cv2.drawMarker(annotated, position, color, cv2.MARKER_CROSS, 9, 1)
        print(f"Sample {sample.sample_index}: edges={len(sample.edges)}, pairs={len(sample.pairs)}")

    # 输出使用独立目录并加序号避让已有文件，原图始终保留。
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else source_path.parent / "caliper_results"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = 0
        while True:
            stem = source_path.stem + "_caliper" + (f"_{suffix}" if suffix else "")
            image_path, data_path = output_dir / (stem + ".png"), output_dir / (stem + ".json")
            if not image_path.exists() and not data_path.exists():
                break
            suffix += 1
        ok, png = cv2.imencode(".png", annotated)
        if not ok:
            raise ValueError("Could not encode result image.")
        png.tofile(str(image_path))
        # 数据记录原图、实际参数及完整找边结果，便于复核和再次运行。
        data_path.write_text(json.dumps({"source_image": str(source_path), "parameters": CONFIG.to_dict(),
                                         "result": result.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError, cv2.error) as exc:
        raise RuntimeError(f"卡尺示例执行失败：{exc}") from exc
    print(f"Found samples: {result.found_sample_count}/{result.sample_count}")
    print(f"Annotated image: {image_path.resolve()}")
    print(f"Measurement data: {data_path.resolve()}")


    if SHOW_WINDOW:
        # 预览副本适配屏幕，保存结果始终保留原始分辨率和原图坐标。
        original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        scale = min(1.0, 1400.0 / (image.shape[1] * 2), 760.0 / image.shape[0])
        size = (max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))))
        left = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(annotated, size, interpolation=cv2.INTER_AREA)
        preview = np.hstack((left, right))
        preview = cv2.copyMakeBorder(preview, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
        cv2.putText(preview, "Original", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(preview, f"Result: {result.found_sample_count}/{result.sample_count} samples",
                    (size[0] + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        window = "Caliper - Original / Result (Esc to close)"
        print("左侧为原图，右侧为找边结果；按Esc/Q或关闭窗口结束。")
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, preview.shape[1], preview.shape[0])
            cv2.imshow(window, preview)
            # 定时检查关闭按钮，避免waitKey(0)在窗口关闭后继续阻塞。
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            # 用户关闭窗口后部分OpenCV版本会提示窗口不存在，此时无需再次关闭。
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
