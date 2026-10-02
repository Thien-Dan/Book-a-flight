"""LỚP 3 — TIÊU CHÍ HOÀN THÀNH KIỂM BẰNG CODE.

Agent nói "xong rồi" không có nghĩa là xong. Harness tự đọc trạng thái thật trong DB
và chấm từng tiêu chí. Chỉ khi TẤT CẢ đạt thì phiên mới được coi là hoàn thành.
"""
from __future__ import annotations

from shared import mock_db as db


def _norm_names(xs):
    return sorted(" ".join(x.split()).casefold() for x in xs or [])


def pick_booking(state: dict) -> dict | None:
    rank = {"TICKETED": 0, "PAID": 1, "HELD": 2, "CANCELLED": 9}
    bs = [db.BOOKINGS[i] for i in state.get("booking_ids") or [] if i in db.BOOKINGS]
    return min(bs, key=lambda b: rank.get(b["status"], 5)) if bs else None


CRITERIA = [
    ("C1", "Có booking hợp lệ trong phiên", lambda b, r, c, s: b is not None and b["status"] != "CANCELLED"),
    ("C2", "Đúng hành trình", lambda b, r, c, s: (b["origin"], b["destination"]) == (r["origin"], r["destination"])),
    ("C3", "Đúng ngày bay", lambda b, r, c, s: b["date"] == r["depart_date"]),
    ("C4", "Đúng hạng ghế", lambda b, r, c, s: b["cabin"] == r["cabin"]),
    ("C5", "Đúng & đủ hành khách", lambda b, r, c, s: _norm_names(b["passengers"]) == _norm_names(r["passengers"])),
    ("C6", "Tổng tiền ≤ ngân sách", lambda b, r, c, s: r.get("budget_vnd") is None or b["total_vnd"] <= r["budget_vnd"]),
    ("C7", "Số điểm dừng ≤ giới hạn", lambda b, r, c, s: b["stops"] <= c["max_stops"]),
    ("C8", "Giờ bay trong khung yêu cầu", lambda b, r, c, s:
        (not r.get("earliest_departure") or b["depart_time"] >= r["earliest_departure"])
        and (not r.get("latest_departure") or b["depart_time"] <= r["latest_departure"])),
    ("C9", "Đã thanh toán đúng số tiền", lambda b, r, c, s: bool(b["payment"]) and b["payment"]["amount_vnd"] == b["total_vnd"]),
    ("C10", "Đã xuất vé cho mọi hành khách", lambda b, r, c, s: b["status"] == "TICKETED" and len(b["tickets"]) == len(b["passengers"])),
    ("C11", "Không còn booking thừa đang treo", lambda b, r, c, s: all(
        db.BOOKINGS[i]["status"] == "CANCELLED" for i in s.get("booking_ids") or [] if i != b["booking_id"] and i in db.BOOKINGS)),
]


def check_completion(state: dict) -> dict:
    b = pick_booking(state)
    r, c = state.get("request") or {}, state.get("constraints") or {}
    results = []
    for cid, desc, fn in CRITERIA:
        try:
            ok = bool(fn(b, r, c, state)) if (b is not None or cid == "C1") else False
        except Exception as e:  # noqa: BLE001
            ok, desc = False, f"{desc} (lỗi kiểm tra: {e})"
        results.append({"id": cid, "criterion": desc, "passed": ok})
    return {
        "done": all(x["passed"] for x in results),
        "booking_id": b["booking_id"] if b else None,
        "criteria": results,
        "failed": [x for x in results if not x["passed"]],
    }
