# -*- coding: utf-8 -*-
"""验证YOLO内存输入、预后处理边界和后端调用协议，无需模型依赖。"""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from Algo.yolo import (ImageTransform, PreprocessConfig, PreprocessMode, YoloBox, YoloDetection,
                       YoloDevice, YoloInferenceConfig, YoloModel, YoloModelConfig, YoloStatus,
                       YoloTask, infer, infer_batch, load_model, merge_same_label_intersections,
                       parse_result, prepare_image, restore_box, restore_boxes, restore_polygon)


def backend_result():
    """模拟后端返回位于预处理图坐标系的框与轮廓。"""
    box = SimpleNamespace(xyxy=np.array([[2, 3, 6, 5]], float), cls=np.array([0]), conf=np.array([0.9]))
    return SimpleNamespace(boxes=[box], names={0: "part"},
                           masks=SimpleNamespace(xy=np.array([[[2, 3], [6, 3], [6, 5]]], float)))


class Backend:
    """记录predict参数并返回每图结果，检查单图/批量是否统一调用。"""
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [backend_result() for _ in kwargs["source"]]


@pytest.mark.parametrize("mode", list(PreprocessMode))
def test_preprocess_modes_and_roundtrip(mode):
    # 奇数尺寸缩放后横纵比例不同，整框仍必须精确还原到输入尺寸。
    source = np.zeros((3, 7, 3), np.uint8)
    processed, transform = prepare_image(source, PreprocessConfig(mode=mode, target_size=10))
    box = YoloBox(transform.pad_left, transform.pad_top, transform.resized_width, transform.resized_height)
    assert restore_box(box, transform) == YoloBox(0, 0, 7, 3)
    assert restore_boxes([box], transform) == [YoloBox(0, 0, 7, 3)]
    polygon = np.array([[box.x, box.y], [box.x + box.width, box.y + box.height]])
    points = restore_polygon(polygon, transform)
    assert [(p.x, p.y) for p in points] == [(0, 0), (7, 3)]
    assert processed.shape == ((3, 7, 3) if mode is PreprocessMode.NONE else (10, 10, 3))
    assert not np.shares_memory(source, processed)
    assert restore_polygon(np.empty((0, 2)), transform) == []


def test_clip_padding_grayscale_and_color():
    # 补边裁剪和禁用裁剪分别验证，BGR转RGB应只影响通道顺序。
    image = np.zeros((2, 4, 3), np.uint8)
    image[:] = (10, 20, 30)
    rgb, transform = prepare_image(image, PreprocessConfig(target_size=4, convert_bgr_to_rgb=True))
    assert rgb[1, 0].tolist() == [30, 20, 10]
    assert restore_box(YoloBox(-5, -5, 20, 20), transform) == YoloBox(0, 0, 4, 2)
    assert restore_box(YoloBox(-5, -5, 20, 20), transform, clip=False).x == -5
    gray, _ = prepare_image(np.ones((2, 4), np.uint8), PreprocessConfig(target_size=4))
    assert gray.shape == (4, 4, 3) and gray[1, 0].tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        prepare_image(image, PreprocessConfig(mode=PreprocessMode.PADDING, target_size=2))


def test_parse_segments_and_merge_are_independent():
    # 轮廓以NumPy容器传入，避免数组真假歧义；合并不能改变原检测或分割框。
    transform = ImageTransform(4, 2, 8, 4, 0, 2, 0, 2)
    detections, segments = parse_result(backend_result(), transform)
    assert detections[0].box == YoloBox(1, 0.5, 2, 1)
    assert segments[0].polygon[0].x == 1
    other = replace(detections[0], confidence=0.8, box=YoloBox(2, 0.5, 2, 1))
    merged = merge_same_label_intersections([other, detections[0]])
    assert merged[0].box == YoloBox(2, 0.5, 1, 1) and merged[0].merged_count == 2
    assert segments[0].box == detections[0].box == YoloBox(1, 0.5, 2, 1)
    raw = backend_result()
    raw.masks.xy = np.empty((0, 3, 2))
    with pytest.raises(ValueError):
        parse_result(raw)
    assert parse_result(raw, include_segments=False)[1] == []
    assert parse_result(SimpleNamespace(boxes=[], masks=None, names={}))[0] == []


@pytest.mark.parametrize("device,expected", [(YoloDevice.AUTO, None), (YoloDevice.CPU, "cpu"), (YoloDevice.CUDA, "2")])
@pytest.mark.parametrize("rgb", [False, True])
def test_single_batch_backend_contract(device, expected, rgb):
    # 后端始终收到BGR；RGB预处理输出由适配层转换，不会重复颠倒模型输入颜色。
    backend = Backend()
    context = YoloModel("model.pt", YoloModelConfig(device=device, gpu_index=2), backend, YoloStatus.OK)
    image = np.zeros((2, 4, 3), np.uint8)
    image[:] = (10, 20, 30)
    original = image.copy()
    config = YoloInferenceConfig(preprocess=PreprocessConfig(target_size=8, convert_bgr_to_rgb=rgb),
                                 confidence_threshold=0.6, merge_same_label=True)
    single = infer(context, image, config)
    batch = infer_batch(context, [image, image.copy()], config)
    assert single.status is YoloStatus.OK
    assert single.detections == batch[0].detections == batch[1].detections
    assert single.raw_detections and batch[0].raw_detections
    assert single.timing.batch_size == 1 and batch[0].timing.batch_size == 2
    call = backend.calls[-1]
    assert call["device"] == expected and call["imgsz"] == 8 and call["conf"] == 0.6
    assert call["source"][0][2, 0].tolist() == [10, 20, 30]
    assert np.array_equal(image, original)
    data = json.loads(json.dumps(single.to_dict()))
    assert data["status"] == "Ok" and "image_path" not in data
    assert data["config"]["preprocess"]["mode"] == "resize_padding"


def test_batch_errors_and_input_boundary():
    # 路径不能穿透到后端；批次个数错误和执行异常返回逐图错误结果。
    backend = Backend()
    context = YoloModel("model.pt", YoloModelConfig(), backend, YoloStatus.OK)
    image = np.zeros((2, 4), np.uint8)
    assert infer_batch(context, []) == []
    for value in ("image.png", None, np.zeros((2, 4), np.float32)):
        with pytest.raises(ValueError):
            infer(context, value)
    with pytest.raises(TypeError):
        infer_batch(context, image)
    assert backend.calls == []
    backend.predict = lambda **kwargs: []
    result = infer_batch(context, [image, image])
    assert len(result) == 2 and all(r.status is YoloStatus.ERROR for r in result)
    def fail(**kwargs):
        """模拟模型运行失败。"""
        raise RuntimeError("backend failure")
    backend.predict = fail
    assert "backend failure" in infer(context, image).message
    unavailable = YoloModel("missing.pt", YoloModelConfig(), message="missing")
    assert infer(unavailable, image).status is YoloStatus.ERROR


def test_loading_uses_local_model_and_task(tmp_path, monkeypatch):
    # 加载测试不访问网络或安装依赖，缺文件不进入后端构造。
    calls = []
    def create(path, task):
        """模拟已加载的后端并检查任务传递。"""
        calls.append((path, task))
        return SimpleNamespace(task=task)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=create))
    path = tmp_path / "model.pt"
    assert load_model(path).status is YoloStatus.ERROR and not calls
    path.write_bytes(b"stub")
    context = load_model(path, YoloModelConfig(task=YoloTask.SEGMENT))
    assert context.status is YoloStatus.OK and calls[-1][1] == "segment"
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *a, **k: SimpleNamespace(task="classify")))
    assert load_model(path).status is YoloStatus.ERROR


@pytest.mark.parametrize("factory,kwargs,error", [
    (YoloModelConfig, {"task": "detect"}, TypeError),
    (YoloModelConfig, {"device": "cpu"}, TypeError),
    (YoloModelConfig, {"gpu_index": -1}, ValueError),
    (PreprocessConfig, {"mode": "padding"}, TypeError),
    (PreprocessConfig, {"target_size": 0}, ValueError),
    (PreprocessConfig, {"target_size": True}, TypeError),
    (PreprocessConfig, {"padding_color": (0, 0, 256)}, ValueError),
    (PreprocessConfig, {"convert_bgr_to_rgb": 1}, TypeError),
    (YoloInferenceConfig, {"confidence_threshold": 2}, ValueError),
    (YoloInferenceConfig, {"iou_threshold": float("nan")}, ValueError),
    (YoloInferenceConfig, {"preprocess": {}}, TypeError),
    (YoloInferenceConfig, {"merge_same_label": 1}, TypeError),
])
def test_config_validation(factory, kwargs, error):
    # 固定选项和数值在构造配置时校验。
    with pytest.raises(error):
        factory(**kwargs)


def test_public_responsibilities():
    # 坐标逆变换只由后处理提供，库不再暴露文件读图函数。
    import Algo.yolo as yolo
    import Algo.yolo.preprocess as preprocess
    assert not hasattr(yolo, "load_image")
    assert not hasattr(preprocess, "restore_box")
    assert restore_box.__module__ == "Algo.yolo.postprocess"
    assert load_model.__module__ == infer.__module__ == "Algo.yolo.model"
