# -*- coding: utf-8 -*-
"""检测和分割例程共用的结果绘图、文件保存与预览，仅供示例使用。"""

import json
from pathlib import Path

import cv2
import numpy as np

from Algo.yolo import YoloResult


def save_and_show(image: np.ndarray, result: YoloResult, source_path: Path,
                  output_dir: Path, show_window: bool):
    """在输入图副本上画检测框和分割轮廓，保存PNG/JSON后可选弹窗。"""
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    annotated = original.copy()
    palette = [(0, 200, 0), (0, 180, 255), (255, 160, 0), (200, 0, 200)]
    # 分割轮廓来自原图坐标；半透明填充只用于显示，不改变算法返回的数据。
    for index, segment in enumerate(result.segments):
        if len(segment.polygon) < 3:
            continue
        points = np.rint([[p.x, p.y] for p in segment.polygon]).astype(np.int32).reshape(-1, 1, 2)
        color = palette[index % len(palette)]
        layer = annotated.copy()
        cv2.fillPoly(layer, [points], color)
        annotated = cv2.addWeighted(layer, 0.3, annotated, 0.7, 0)
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
    for index, detection in enumerate(result.detections):
        box = detection.box
        start = (int(round(box.x)), int(round(box.y)))
        end = (int(round(box.x + box.width)), int(round(box.y + box.height)))
        color = palette[index % len(palette)]
        cv2.rectangle(annotated, start, end, color, 2)
        # OpenCV内置字体无法渲染中文类别，图中用class_id兜底，JSON保留完整标签。
        label = detection.label if detection.label.isascii() else f"class_{detection.class_id}"
        cv2.putText(annotated, f"{label} {detection.confidence:.3f}",
                    (max(0, start[0]), max(15, start[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        print(f"Detection {index}: label={detection.label}, confidence={detection.confidence:.3f}, box={box}")
    for index, segment in enumerate(result.segments):
        print(f"Segment {index}: label={segment.label}, points={len(segment.polygon)}")

    # 每次保存独立结果，避免覆盖输入图片或前次调试记录。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        stem = source_path.stem + "_result" + (f"_{suffix}" if suffix else "")
        image_path, data_path = output_dir / (stem + ".png"), output_dir / (stem + ".json")
        if not image_path.exists() and not data_path.exists():
            break
        suffix += 1
    ok, png = cv2.imencode(".png", annotated)
    if not ok:
        raise RuntimeError("Could not encode result image.")
    png.tofile(str(image_path))
    data_path.write_text(json.dumps({"source_image": str(source_path), "result": result.to_dict()},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detections: {len(result.detections)}, segments: {len(result.segments)}")
    print(f"Annotated image: {image_path.resolve()}")
    print(f"Measurement data: {data_path.resolve()}")

    if show_window:
        # 仅缩小预览副本，原尺寸标记图和精确坐标已经保存到文件。
        scale = min(1.0, 1400.0 / (image.shape[1] * 2), 760.0 / image.shape[0])
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        preview = np.hstack((cv2.resize(original, size), cv2.resize(annotated, size)))
        preview = cv2.copyMakeBorder(preview, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
        cv2.putText(preview, "Original", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(preview, f"Result: {len(result.detections)} boxes / {len(result.segments)} segments",
                    (size[0] + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        window = "YOLO - Original / Result (Esc to close)"
        print("左侧原图，右侧结果；按Esc/Q或关闭窗口结束。")
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, preview.shape[1], preview.shape[0])
            cv2.imshow(window, preview)
            while True:
                if cv2.waitKey(30) & 0xFF in (27, ord("q"), ord("Q")):
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    return image_path, data_path
