"""POST /loss-allocation 的端到端测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    """与 test_evaluate 相同的三层小树：

    source --s1(0.1Ω, 20A)--> a --s2(0.2Ω, 10A)--> b
                              |
                              +--s3(0.05Ω, 15A)--> c

    负载：a=2A, b=5A, c=8A；段电流：s1=15, s2=5, s3=8
    线损：s1=15²×0.1=22.5, s2=5²×0.2=5.0, s3=8²×0.05=3.2，总线损 30.7
    归属：负载0(a,2A) 路径[s1]    -> 2×15×0.1 = 3.0
          负载1(b,5A) 路径[s2,s1] -> 5×5×0.2 + 5×15×0.1 = 5.0+7.5 = 12.5
          负载2(c,8A) 路径[s3,s1] -> 8×8×0.05 + 8×15×0.1 = 3.2+12.0 = 15.2
    """
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 15.0},
        ],
        "loads": [
            {"node": "a", "current": 2.0},
            {"node": "b", "current": 5.0},
            {"node": "c", "current": 8.0},
        ],
    }
    payload.update(overrides)
    return payload


# ---------- 逐线段线损与总线损 ----------


def test_segment_losses_and_total():
    resp = client.post("/loss-allocation", json=base_payload())
    assert resp.status_code == 200
    data = resp.json()

    segments = {s["segment_id"]: s for s in data["segments"]}
    assert [s["segment_id"] for s in data["segments"]] == ["s1", "s2", "s3"]  # 保持输入顺序
    assert segments["s1"]["current"] == pytest.approx(15.0)
    assert segments["s1"]["resistance"] == pytest.approx(0.1)
    assert segments["s1"]["loss"] == pytest.approx(22.5)
    assert segments["s2"]["loss"] == pytest.approx(5.0)
    assert segments["s3"]["loss"] == pytest.approx(3.2)
    assert data["total_loss"] == pytest.approx(30.7)
    assert sum(s["loss"] for s in data["segments"]) == pytest.approx(data["total_loss"])


# ---------- 逐负载归属 ----------


def test_per_load_allocation():
    resp = client.post("/loss-allocation", json=base_payload())
    assert resp.status_code == 200
    data = resp.json()

    assert [item["index"] for item in data["loads"]] == [0, 1, 2]
    allocations = {item["index"]: item["allocated_loss"] for item in data["loads"]}
    assert allocations[0] == pytest.approx(3.0)
    assert allocations[1] == pytest.approx(12.5)
    assert allocations[2] == pytest.approx(15.2)
    # 逐负载归属之和与总线损一致
    assert sum(allocations.values()) == pytest.approx(data["total_loss"])


def test_source_node_load_allocated_zero():
    payload = base_payload()
    payload["loads"].append({"node": "source", "current": 4.0})
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # source 负载不经过任何线段，归属为零，且不影响各段线损与总线损
    source_load = data["loads"][3]
    assert source_load["node"] == "source"
    assert source_load["current"] == pytest.approx(4.0)
    assert source_load["allocated_loss"] == pytest.approx(0.0)
    assert data["total_loss"] == pytest.approx(30.7)
    assert sum(item["allocated_loss"] for item in data["loads"]) == pytest.approx(30.7)


def test_multiple_loads_same_node_listed_separately():
    payload = base_payload()
    payload["loads"].append({"node": "b", "current": 1.0})  # b: 5 + 1 = 6A
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # s2 电流 6A、s1 电流 16A：总线损 = 16²×0.1 + 6²×0.2 + 8²×0.05 = 25.6+7.2+3.2 = 36.0
    assert data["total_loss"] == pytest.approx(36.0)
    # 同节点 b 的两条负载分别列出（序号 1 与 3）
    assert [(item["index"], item["node"]) for item in data["loads"]] == [
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "b"),
    ]
    allocations = {item["index"]: item["allocated_loss"] for item in data["loads"]}
    assert allocations[0] == pytest.approx(2 * 16 * 0.1)  # 3.2
    assert allocations[1] == pytest.approx(5 * 6 * 0.2 + 5 * 16 * 0.1)  # 14.0
    assert allocations[2] == pytest.approx(8 * 8 * 0.05 + 8 * 16 * 0.1)  # 16.0
    assert allocations[3] == pytest.approx(1 * 6 * 0.2 + 1 * 16 * 0.1)  # 2.8
    assert sum(allocations.values()) == pytest.approx(36.0)


def test_empty_loads_zero_loss():
    resp = client.post("/loss-allocation", json=base_payload(loads=[]))
    assert resp.status_code == 200
    data = resp.json()

    assert data["loads"] == []
    assert data["total_loss"] == pytest.approx(0.0)
    assert all(s["current"] == pytest.approx(0.0) for s in data["segments"])
    assert all(s["loss"] == pytest.approx(0.0) for s in data["segments"])


def test_zero_resistance_segment_zero_loss():
    payload = base_payload()
    payload["segments"][1]["resistance"] = 0.0  # s2 无电阻
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    segments = {s["segment_id"]: s for s in data["segments"]}
    assert segments["s2"]["loss"] == pytest.approx(0.0)
    # 负载1(b,5A) 仅在 s1 上承担：5×15×0.1 = 7.5
    assert data["loads"][1]["allocated_loss"] == pytest.approx(7.5)
    assert data["total_loss"] == pytest.approx(25.7)


def test_allocation_consistent_with_total_under_rounding():
    # 非有限小数电流：计算与汇总使用未舍入值，展示舍入差在 1e-6 量级内
    payload = base_payload(
        loads=[
            {"node": "a", "current": 0.1},
            {"node": "b", "current": 1 / 3},
            {"node": "c", "current": 0.7},
            {"node": "c", "current": 0.2},
        ]
    )
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    allocated_sum = sum(item["allocated_loss"] for item in data["loads"])
    segment_sum = sum(s["loss"] for s in data["segments"])
    assert allocated_sum == pytest.approx(data["total_loss"], abs=2e-6)
    assert segment_sum == pytest.approx(data["total_loss"], abs=2e-6)


def test_segment_ids_auto_generated():
    payload = base_payload()
    for seg in payload["segments"]:
        del seg["id"]
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 200
    ids = [s["segment_id"] for s in resp.json()["segments"]]
    assert ids == ["seg-1", "seg-2", "seg-3"]


# ---------- 校验与错误响应（复用 /evaluate） ----------


def test_negative_resistance_rejected():
    payload = base_payload()
    payload["segments"][0]["resistance"] = -0.1
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert any("电阻不能为负" in d["message"] for d in error["details"])


def test_negative_load_rejected():
    payload = base_payload()
    payload["loads"][0]["current"] = -1.0
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 422
    assert any("负载电流不能为负" in d["message"] for d in resp.json()["error"]["details"])


def test_disconnected_segment_rejected():
    payload = base_payload()
    payload["segments"].append(
        {"parent": "x", "child": "y", "resistance": 0.1, "rated_current": 10.0}
    )
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("不连通" in d["message"] for d in error["details"])


def test_unknown_load_node_rejected():
    payload = base_payload()
    payload["loads"].append({"node": "ghost", "current": 1.0})
    resp = client.post("/loss-allocation", json=payload)
    assert resp.status_code == 400
    assert any("不存在" in d["message"] for d in resp.json()["error"]["details"])
