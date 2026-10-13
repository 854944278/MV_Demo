# -*- coding: utf-8 -*-
"""用已知边界的合成图验证卡尺坐标、采样平均和选择语义。"""

import json

import numpy as np
import pytest

from Algo.caliper import (CaliperConfig, CaliperResult, Edge, EdgePair, Point, ScanDirection,
                          EdgeMode, EdgeSelection, EdgePolarity, PairPolarity, caliper_find_edges)


@pytest.mark.parametrize("direction,axis,polarity", [
    (ScanDirection.LEFT_TO_RIGHT, "x", EdgePolarity.DARK_TO_BRIGHT),
    (ScanDirection.RIGHT_TO_LEFT, "x", EdgePolarity.BRIGHT_TO_DARK),
    (ScanDirection.TOP_TO_BOTTOM, "y", EdgePolarity.DARK_TO_BRIGHT),
    (ScanDirection.BOTTOM_TO_TOP, "y", EdgePolarity.BRIGHT_TO_DARK),
])
def test_four_directions(direction, axis, polarity):
    # 同一物理边界的坐标不因反向扫描而改变，极性则随扫描方向反转。
    image = np.zeros((7, 12), np.uint8)
    image[:, 4:] = 200
    if axis == "y":
        image = image.T
    result = caliper_find_edges(image, CaliperConfig(direction=direction, edge_polarity=polarity, sample_count=3))
    assert result.found_sample_count == 3
    assert [(s.strip_start, s.strip_stop) for s in result.samples] == [(0, 3), (3, 5), (5, 7)]
    for sample, center in zip(result.samples, [1, 3.5, 5.5]):
        edge, = sample.edges
        assert getattr(edge, axis) == 3.5
        assert getattr(edge, "y" if axis == "x" else "x") == center
        assert edge.strength == 200
        assert edge.polarity == polarity
    # 返回字段应可直接序列化，方便未来项目跨语言调用。
    json.dumps(result.to_dict())


@pytest.mark.parametrize("selection,expected", [
    (EdgeSelection.ALL, [1.5, 4.5, 7.5]), (EdgeSelection.FIRST, [1.5]), (EdgeSelection.LAST, [7.5]), (EdgeSelection.STRONGEST, [4.5]),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_single_selection(selection, expected, reverse):
    # 三个相同极性的边缘，中间边缘最强；反向后first和last交换物理位置。
    image = np.array([[0, 0, 40, 40, 40, 140, 140, 140, 200, 200]], np.uint8)
    result = caliper_find_edges(image, CaliperConfig(direction=ScanDirection.RIGHT_TO_LEFT if reverse else ScanDirection.LEFT_TO_RIGHT, selection=selection))
    if reverse and selection == EdgeSelection.ALL:
        expected = expected[::-1]
    elif reverse and selection in (EdgeSelection.FIRST, EdgeSelection.LAST):
        expected = [7.5] if selection == EdgeSelection.FIRST else [1.5]
    assert [e.x for e in result.samples[0].edges] == expected


@pytest.mark.parametrize("transpose", [False, True])
def test_reduce_uses_all_pixels_and_preserves_fraction(transpose):
    # 五行的阶跃幅值平均为36.2，逐行采样则保留101和80两个局部阶跃。
    image = np.zeros((5, 8), np.uint8)
    image[:, 3:] = np.array([0, 101, 0, 0, 80], np.uint8)[:, None]
    if transpose:
        image = image.T
    direction = ScanDirection.TOP_TO_BOTTOM if transpose else ScanDirection.LEFT_TO_RIGHT
    averaged = caliper_find_edges(image, CaliperConfig(direction=direction, min_contrast=0))
    assert averaged.samples[0].edges[0].strength == pytest.approx(36.2)
    split = caliper_find_edges(image, CaliperConfig(direction=direction, sample_count=5, min_contrast=50))
    assert [s.found for s in split.samples] == [False, True, False, False, True]


@pytest.mark.parametrize("direction", [ScanDirection.LEFT_TO_RIGHT, ScanDirection.RIGHT_TO_LEFT, ScanDirection.TOP_TO_BOTTOM, ScanDirection.BOTTOM_TO_TOP])
@pytest.mark.parametrize("dark", [False, True])
def test_pair_coordinates_and_width(direction, dark):
    # 已知宽度为5像素的条带，反向仅交换first/second，不改变中心和宽度。
    image = np.zeros((4, 12), np.uint8)
    image[:, 3:8] = 200
    if dark:
        image = 200 - image
    vertical = direction in (ScanDirection.TOP_TO_BOTTOM, ScanDirection.BOTTOM_TO_TOP)
    if vertical:
        image = image.T
    result = caliper_find_edges(image, CaliperConfig(direction=direction, edge_mode=EdgeMode.PAIR, sample_count=2,
                        pair_polarity=PairPolarity.DARK_STRIP if dark else PairPolarity.BRIGHT_STRIP, min_width=5, max_width=5))
    for sample in result.samples:
        pair, = sample.pairs
        axis = "y" if vertical else "x"
        assert pair.width == 5
        assert getattr(pair.center, axis) == 5
        assert pair.strength == 200
        expected = [7.5, 2.5] if direction in (ScanDirection.RIGHT_TO_LEFT, ScanDirection.BOTTOM_TO_TOP) else [2.5, 7.5]
        assert [getattr(pair.first, axis), getattr(pair.second, axis)] == expected
        assert sample.edges == []


@pytest.mark.parametrize("selection,centers", [
    (EdgeSelection.ALL, [2.5, 6.5]), (EdgeSelection.FIRST, [2.5]), (EdgeSelection.LAST, [6.5]), (EdgeSelection.STRONGEST, [6.5]),
])
def test_pair_selection_uses_weaker_side(selection, centers):
    # 第一对强度为min(200,40)，第二对为min(90,90)，最强应选择第二对。
    image = np.array([[0, 0, 200, 200, 160, 160, 250, 250, 160, 160]], np.uint8)
    result = caliper_find_edges(image, CaliperConfig(edge_mode=EdgeMode.PAIR, pair_polarity=PairPolarity.BRIGHT_STRIP, selection=selection))
    assert [p.center.x for p in result.samples[0].pairs] == centers


def test_pairs_share_boundary_and_do_not_bridge_weak_edges():
    # both模式允许亮暗条带共享中间边缘。
    image = np.array([[0, 0, 100, 100, 0, 0, 100, 100]], np.uint8)
    pairs = caliper_find_edges(image, CaliperConfig(edge_mode=EdgeMode.PAIR)).samples[0].pairs
    assert [p.polarity for p in pairs] == [PairPolarity.BRIGHT_STRIP, PairPolarity.DARK_STRIP]
    assert pairs[0].second == pairs[1].first
    # 强边缘之间夹着弱响应时，不能跳过弱响应构造一个大边缘对。
    weak = np.array([[0, 0, 200, 200, 195, 195, 200, 200, 0, 0]], np.uint8)
    assert not caliper_find_edges(weak, CaliperConfig(edge_mode=EdgeMode.PAIR, min_contrast=10)).found
    assert not caliper_find_edges(image, CaliperConfig(edge_mode=EdgeMode.PAIR, min_width=3)).found
    assert not caliper_find_edges(image, CaliperConfig(edge_mode=EdgeMode.PAIR, max_width=1)).found


def test_plateau_narrow_strip_smoothing_and_empty_result():
    # 等斜率缓坡的多个梯度像素合并为一个平台峰。
    ramp = np.array([[0, 0, 20, 40, 60, 60]], np.uint8)
    edges = caliper_find_edges(ramp).samples[0].edges
    assert len(edges) == 1 and edges[0].x == 2.5
    narrow = np.array([[0, 200, 0]], np.uint8)
    assert caliper_find_edges(narrow, CaliperConfig(edge_mode=EdgeMode.PAIR)).samples[0].pairs[0].width == 1
    step = np.zeros((3, 30), np.uint8)
    step[:, 15:] = 200
    smoothed = caliper_find_edges(step, CaliperConfig(smooth_sigma=1)).samples[0].edges
    assert len(smoothed) == 1
    assert smoothed[0].x == pytest.approx(14.5)
    assert 0 < smoothed[0].strength < 200
    assert caliper_find_edges(np.zeros((2, 4), np.uint8), CaliperConfig(min_contrast=0)).found is False
    assert caliper_find_edges(step, CaliperConfig(edge_polarity=EdgePolarity.BRIGHT_TO_DARK)).found is False


def test_bgr_noncontiguous_and_tie():
    # 支持非连续ROI；强度并列时按扫描顺序取最先遇到者。
    image = np.array([[0, 0, 100, 100, 0, 0]], np.uint8)
    bgr = np.repeat(np.repeat(image[..., None], 3, axis=2), 3, axis=0)
    assert caliper_find_edges(bgr[::2], CaliperConfig(selection=EdgeSelection.STRONGEST)).samples[0].edges[0].x == 1.5
    assert caliper_find_edges(image, CaliperConfig(direction=ScanDirection.RIGHT_TO_LEFT, selection=EdgeSelection.STRONGEST)).samples[0].edges[0].x == 3.5


@pytest.mark.parametrize("kwargs", [
    {"direction": "diagonal"}, {"edge_mode": "invalid"}, {"selection": "nearest"},
    {"edge_polarity": PairPolarity.BRIGHT_STRIP}, {"pair_polarity": EdgePolarity.DARK_TO_BRIGHT},
    {"sample_count": 0}, {"sample_count": 4}, {"sample_count": 1.5}, {"sample_count": True},
    {"min_contrast": -1}, {"min_contrast": float("nan")}, {"smooth_sigma": float("inf")},
    {"min_width": -1}, {"min_width": 5, "max_width": 4}, {"max_width": "5"},
])
def test_invalid_parameters(kwargs):
    # 非法参数明确抛出异常，不静默回退为其他搜索策略。
    with pytest.raises((TypeError, ValueError)):
        caliper_find_edges(np.zeros((3, 8), np.uint8), CaliperConfig(**kwargs))


@pytest.mark.parametrize("image", [None, np.zeros((0, 5), np.uint8), np.zeros((3, 5), np.float32),
                                        np.zeros((3, 5, 4), np.uint8), np.zeros((3, 1), np.uint8)])
def test_invalid_images(image):
    # 限定图像类型和扫描长度，避免强度单位不一致或空剖面。
    with pytest.raises(ValueError):
        caliper_find_edges(image)


def test_old_exports_removed():
    # Algo公开入口只保留统一找边，原工程算子不参与本测试。
    import Algo.caliper as caliper
    assert "caliper_find_edges" in caliper.__all__
    assert "CaliperConfig" in caliper.__all__
    for name in ("measure_edge_line", "measure_line_center", "measure_paired_edges"):
        assert not hasattr(caliper, name)


@pytest.mark.parametrize("kwargs", [
    {"direction": "left_to_right"}, {"edge_mode": "single"}, {"selection": "all"},
    {"edge_polarity": "both"}, {"pair_polarity": "both"},
    {"direction": EdgeSelection.FIRST}, {"sample_count": np.bool_(True)},
    {"min_contrast": np.bool_(False)},
])
def test_config_rejects_strings_and_wrong_types(kwargs):
    # 即便字符串拼写正确，也必须显式使用对应枚举，不能绕过类型约束。
    with pytest.raises(TypeError):
        CaliperConfig(**kwargs)


def test_config_is_validated_immutable_and_reusable():
    # 修改只读配置应重新构造，replace也会执行校验；多次运行结果相互独立。
    from dataclasses import FrozenInstanceError, replace
    config = CaliperConfig(sample_count=np.int64(2), min_contrast=np.float32(8))
    with pytest.raises(FrozenInstanceError):
        config.sample_count = 0
    with pytest.raises(ValueError):
        replace(config, min_width=5, max_width=2)
    json.dumps(config.to_dict())
    assert replace(config, selection=EdgeSelection.FIRST).selection is EdgeSelection.FIRST
    image = np.array([[0, 100, 0], [0, 100, 0]], np.uint8)
    first = caliper_find_edges(image, config)
    second = caliper_find_edges(image, config)
    first.samples[0].edges.clear()
    assert len(second.samples[0].edges) == 2
    with pytest.raises(TypeError):
        caliper_find_edges(image, config.to_dict())


def test_typed_result_and_json_snapshot():
    # 算法内保留枚举及嵌套类型；JSON快照转换为字符串且不共享可变列表。
    image = np.array([[0, 0, 200, 200, 0, 0]], np.uint8)
    config = CaliperConfig(edge_mode=EdgeMode.PAIR, pair_polarity=PairPolarity.BRIGHT_STRIP)
    result = caliper_find_edges(image, config)
    assert isinstance(result, CaliperResult)
    pair = result.samples[0].pairs[0]
    assert isinstance(pair, EdgePair) and isinstance(pair.first, Edge) and isinstance(pair.center, Point)
    assert pair.first.polarity is EdgePolarity.DARK_TO_BRIGHT
    assert pair.polarity is PairPolarity.BRIGHT_STRIP
    snapshot = json.loads(json.dumps(result.to_dict()))
    assert snapshot["config"]["edge_mode"] == "pair"
    assert snapshot["samples"][0]["pairs"][0]["first"]["polarity"] == "dark_to_bright"
    assert snapshot["found"] and snapshot["found_sample_count"] == 1
    snapshot["samples"].clear()
    assert len(result.samples) == 1
    empty = caliper_find_edges(np.zeros((2, 4), np.uint8))
    assert not empty.found and empty.found_sample_count == 0


@pytest.mark.parametrize("mode", [EdgeMode.SINGLE, EdgeMode.PAIR])
def test_pycharm_example_typed_output(tmp_path, monkeypatch, mode):
    # 验证实际读图、配置、绘图和JSON链路；GUI用替身检查预览并避免弹出测试窗口。
    import cv2
    import runpy
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "examples" / "caliper_example.py"
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(script), run_name="caliper_example_test")
    settings = namespace["main"].__globals__
    source = tmp_path / "sample.png"
    image = np.zeros((20, 40), np.uint8)
    image[:, 10:25] = 200
    cv2.imencode(".png", image)[1].tofile(str(source))
    settings.update(IMAGE_PATH=str(source), CONFIG=CaliperConfig(edge_mode=mode, sample_count=4))
    shown = []
    for name in ("namedWindow", "resizeWindow", "destroyWindow"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, preview: shown.append(preview))
    monkeypatch.setattr(cv2, "waitKey", lambda *args: 27)
    namespace["main"]()
    assert len(shown) == 1 and shown[0].shape == (55, 80, 3)
    data = json.loads(next((tmp_path / "caliper_results").glob("*.json")).read_text(encoding="utf-8"))
    assert data["parameters"]["edge_mode"] == mode.value
    assert data["result"]["found_sample_count"] == 4
    sample = data["result"]["samples"][0]
    if mode is EdgeMode.PAIR:
        assert sample["pairs"][0]["width"] == 15
    else:
        assert [edge["x"] for edge in sample["edges"]] == [9.5, 24.5]
