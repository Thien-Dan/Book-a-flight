"""LỚP 1 — RÀNG BUỘC LÀ DỮ LIỆU.

- Policy hệ thống nằm trong config/policy.json.
- Yêu cầu của khách được trích thành dữ liệu có cấu trúc (TripRequest).
- Luật kiểm tra tool (tool_constraints) là các bản ghi {field, op, ref|value}; file này chỉ là
  bộ thông dịch chung -> thêm/bớt luật không cần sửa code.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from shared import mock_db as db

POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "policy.json"


def load_policy(overrides: dict | None = None) -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        node = policy
        *path, last = dotted.split(".")
        for k in path:
            node = node[k]
        node[last] = value
    return policy


class TripRequest(BaseModel):
    """Yêu cầu đặt vé của khách, trích từ hội thoại."""
    origin: Optional[str] = Field(None, description="Mã IATA điểm đi, VD HAN, SGN, DAD")
    destination: Optional[str] = Field(None, description="Mã IATA điểm đến")
    depart_date: Optional[str] = Field(None, description="Ngày bay YYYY-MM-DD (tự quy đổi 'mai', 'thứ 6 tuần sau'...)")
    passengers: list[str] = Field(default_factory=list, description="Họ tên đầy đủ từng hành khách (nếu khách có nêu)")
    num_passengers: Optional[int] = Field(None, description="Số hành khách khách nói tới")
    cabin: str = Field("economy", description="'economy' hoặc 'business'")
    budget_vnd: Optional[int] = Field(None, description="Ngân sách TỔNG cho cả đoàn, đơn vị VND (3 triệu = 3000000)")
    max_stops: Optional[int] = Field(None, description="Số điểm dừng tối đa, 'bay thẳng' = 0")
    avoid_airlines: list[str] = Field(default_factory=list, description="Mã hãng muốn tránh: VN, VJ, QH, VU")
    earliest_departure: Optional[str] = Field(None, description="Giờ cất cánh sớm nhất HH:MM")
    latest_departure: Optional[str] = Field(None, description="Giờ cất cánh muộn nhất HH:MM")
    preference: Optional[str] = Field(None, description="Ưu tiên khác: rẻ nhất, sớm nhất, hãng X...")


def normalize_request(req: TripRequest) -> dict:
    r = req.model_dump()
    for k in ("origin", "destination"):
        r[k] = r[k].strip().upper() if r[k] else None
    r["passengers"] = [re.sub(r"\s+", " ", p).strip().title() for p in r["passengers"] if p and p.strip()]
    if not r["num_passengers"]:
        r["num_passengers"] = len(r["passengers"]) or None
    r["cabin"] = (r["cabin"] or "economy").lower()
    r["avoid_airlines"] = [a.upper() for a in r["avoid_airlines"]]
    return r


def validate_request(r: dict, policy: dict) -> list[dict]:
    """Kiểm tra yêu cầu so với policy. severity: missing (cần hỏi khách) | violation (vi phạm luật)."""
    lim = policy["limits"]
    issues = []
    for f, label in [("origin", "điểm đi"), ("destination", "điểm đến"), ("depart_date", "ngày bay")]:
        if not r.get(f):
            issues.append({"field": f, "severity": "missing", "message": f"Thiếu {label}"})
    for f in ("origin", "destination"):
        if r.get(f) and r[f] not in db.AIRPORTS:
            issues.append({"field": f, "severity": "violation", "message": f"Sân bay {r[f]} chưa được hỗ trợ"})
    if r.get("origin") and r["origin"] == r.get("destination"):
        issues.append({"field": "destination", "severity": "violation", "message": "Điểm đi trùng điểm đến"})
    if r.get("depart_date"):
        try:
            d = datetime.strptime(r["depart_date"], "%Y-%m-%d").date()
            delta = (d - date.today()).days
            if delta < 0:
                issues.append({"field": "depart_date", "severity": "violation", "message": "Ngày bay đã qua"})
            elif delta > lim["max_days_in_advance"]:
                issues.append({"field": "depart_date", "severity": "violation",
                               "message": f"Chỉ đặt trước tối đa {lim['max_days_in_advance']} ngày"})
        except ValueError:
            issues.append({"field": "depart_date", "severity": "violation", "message": "Ngày bay sai định dạng"})
    n, names = r.get("num_passengers") or 0, r.get("passengers") or []
    if not names:
        issues.append({"field": "passengers", "severity": "missing", "message": "Thiếu họ tên hành khách"})
    elif n and len(names) < n:
        issues.append({"field": "passengers", "severity": "missing",
                       "message": f"Khách nói {n} người nhưng mới có {len(names)} họ tên"})
    if max(n, len(names)) > lim["max_passengers"]:
        issues.append({"field": "passengers", "severity": "violation",
                       "message": f"Tối đa {lim['max_passengers']} khách/booking"})
    if r.get("cabin") not in lim["allowed_cabins"]:
        issues.append({"field": "cabin", "severity": "violation", "message": f"Hạng {r.get('cabin')} không được phép"})
    return issues


def effective_constraints(r: dict, policy: dict) -> dict:
    lim = policy["limits"]
    stops = lim["max_stops"] if r.get("max_stops") is None else min(r["max_stops"], lim["max_stops"])
    return {
        "max_stops": stops,
        "budget_vnd": r.get("budget_vnd"),
        "avoid_airlines": r.get("avoid_airlines") or [],
        "time_window": [r.get("earliest_departure"), r.get("latest_departure")],
        "cabin": r.get("cabin"),
        "max_payment_vnd": lim["max_payment_vnd"],
    }


# ------------------------- bộ thông dịch luật tool -------------------------

def _get(ctx: dict, path: str) -> Any:
    node: Any = ctx
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip().casefold()
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def _apply(op: str, a: Any, b: Any) -> bool:
    a, b = _norm(a), _norm(b)
    if op == "==":
        return str(a) == str(b) if isinstance(a, str) or isinstance(b, str) else a == b
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    if op == "in":
        return a in (b or [])
    if op == "not_in":
        return a not in (b or [])
    if op == "same_set":
        return sorted(a or []) == sorted(b or [])
    raise ValueError(f"op không hỗ trợ: {op}")


def build_tool_context(tool: str, args: dict, state: dict) -> dict:
    ctx = {
        "args": copy.deepcopy(args),
        "request": state.get("request") or {},
        "constraints": state.get("constraints") or {},
        "policy": state.get("policy") or {},
        "session": {"booking_ids": state.get("booking_ids") or []},
        "computed": {},
    }
    if tool == "hold_booking":
        f = db.FLIGHTS.get(args.get("flight_id", ""))
        if f:
            ctx["flight"] = f
            price = f["prices_vnd"].get(args.get("cabin", ""), 0)
            ctx["computed"]["total_vnd"] = price * len(args.get("passengers") or [])
    if "booking_id" in args:
        ctx["booking"] = db.BOOKINGS.get(args["booking_id"])
    return ctx


def check_tool_constraints(tool: str, args: dict, state: dict) -> list[dict]:
    """Trả về danh sách kết quả từng luật áp dụng cho tool này: {id, passed, message, actual, expected}."""
    ctx = build_tool_context(tool, args, state)
    results = []
    for rule in state["policy"]["tool_constraints"]:
        if rule["tool"] != tool:
            continue
        actual = _get(ctx, rule["field"])
        expected = rule["value"] if "value" in rule else _get(ctx, rule["ref"])
        # Luật không áp dụng khi khách không đặt ràng buộc đó (ref rỗng) hoặc chưa có dữ liệu để so
        if expected is None or actual is None or (expected == [] and rule["op"] == "not_in"):
            results.append({"id": rule["id"], "passed": True, "skipped": True, "message": rule["message"]})
            continue
        try:
            passed = _apply(rule["op"], actual, expected)
        except TypeError:
            passed = False
        results.append({"id": rule["id"], "passed": passed, "skipped": False, "message": rule["message"],
                        "actual": actual, "op": rule["op"], "expected": expected})
    return results
