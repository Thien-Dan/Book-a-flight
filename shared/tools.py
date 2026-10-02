"""Tool mockup cho agent đặt vé máy bay.

Tool chỉ làm nghiệp vụ "thô" (như API thật của hãng). Mọi ràng buộc theo yêu cầu khách,
quyền hạn, phê duyệt... KHÔNG nằm ở đây mà nằm ở lớp harness (harness/guard.py).
"""
from __future__ import annotations

import json
import random
import string
from datetime import datetime

from langchain_core.tools import tool

from shared import mock_db as db


def _ok(**data) -> str:
    return json.dumps({"ok": True, **data}, ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)


@tool
def search_flights(origin: str, destination: str, date: str, passengers: int = 1, cabin: str = "economy") -> str:
    """Tìm chuyến bay một chiều. origin/destination là mã IATA (VD: HAN, SGN), date dạng YYYY-MM-DD,
    passengers là số khách, cabin là 'economy' hoặc 'business'. Trả về danh sách chuyến còn đủ ghế."""
    origin, destination = origin.upper(), destination.upper()
    if origin not in db.AIRPORTS or destination not in db.AIRPORTS:
        return _err(f"Mã sân bay không hỗ trợ. Hỗ trợ: {', '.join(db.AIRPORTS)}")
    flights = db.generate_flights(origin, destination, date)
    result = []
    for f in flights:
        if f["seats_left"].get(cabin, 0) < passengers:
            continue
        result.append({
            "flight_id": f["flight_id"], "airline": f["airline"], "airline_name": f["airline_name"],
            "depart_time": f["depart_time"], "arrive_time": f["arrive_time"], "stops": f["stops"], "via": f["via"],
            "unit_price_vnd": f["prices_vnd"][cabin], "total_vnd": f["prices_vnd"][cabin] * passengers,
            "seats_left": f["seats_left"][cabin],
        })
    return _ok(count=len(result), cabin=cabin, flights=result)


@tool
def get_flight_details(flight_id: str) -> str:
    """Xem chi tiết một chuyến bay (hành lý, điều kiện hoàn/đổi, giá các hạng). Phải search_flights trước."""
    f = db.FLIGHTS.get(flight_id)
    if not f:
        return _err("Không tìm thấy flight_id. Hãy gọi search_flights trước.")
    info = db.AIRLINES[f["airline"]]
    return _ok(flight={**f, "baggage": info["baggage"], "refund_policy": info["refund"]})


@tool
def hold_booking(flight_id: str, cabin: str, passengers: list[str]) -> str:
    """Giữ chỗ (chưa thanh toán) cho danh sách họ tên hành khách trên một chuyến. Trả về booking_id và tổng tiền.
    Giữ chỗ hết hạn sau 20 phút."""
    f = db.FLIGHTS.get(flight_id)
    if not f:
        return _err("Không tìm thấy flight_id. Hãy gọi search_flights trước.")
    if cabin not in f["prices_vnd"]:
        return _err("Hạng ghế không hợp lệ")
    if not passengers:
        return _err("Cần ít nhất 1 hành khách")
    if f["seats_left"][cabin] < len(passengers):
        return _err(f"Không đủ ghế: còn {f['seats_left'][cabin]}")
    b = db.create_hold(f, cabin, passengers)
    return _ok(booking=b)


@tool
def process_payment(booking_id: str, amount_vnd: int) -> str:
    """Thanh toán booking bằng phương thức mặc định của khách (thẻ ****4242). amount_vnd phải bằng tổng booking."""
    b = db.BOOKINGS.get(booking_id)
    if not b:
        return _err("Không tìm thấy booking")
    if b["status"] != "HELD":
        return _err(f"Booking đang ở trạng thái {b['status']}, không thể thanh toán")
    if int(amount_vnd) != b["total_vnd"]:
        return _err(f"Sai số tiền: cần {b['total_vnd']}")
    b["status"] = "PAID"
    b["payment"] = {
        "payment_id": "PAY" + "".join(random.choices(string.digits, k=8)),
        "amount_vnd": b["total_vnd"], "method": "VISA ****4242", "paid_at": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
    }
    return _ok(booking_id=booking_id, status=b["status"], payment=b["payment"])


@tool
def issue_ticket(booking_id: str) -> str:
    """Xuất vé điện tử cho booking đã thanh toán. Trả về số vé cho từng hành khách."""
    b = db.BOOKINGS.get(booking_id)
    if not b:
        return _err("Không tìm thấy booking")
    if b["status"] == "TICKETED":
        return _ok(booking_id=booking_id, status=b["status"], tickets=b["tickets"])
    if b["status"] != "PAID":
        return _err("Booking chưa thanh toán")
    b["tickets"] = [{"passenger": p, "ticket_no": "738-" + "".join(random.choices(string.digits, k=10))} for p in b["passengers"]]
    b["status"] = "TICKETED"
    return _ok(booking_id=booking_id, status=b["status"], tickets=b["tickets"])


@tool
def get_booking(booking_id: str) -> str:
    """Xem trạng thái hiện tại của một booking."""
    b = db.BOOKINGS.get(booking_id)
    return _ok(booking=b) if b else _err("Không tìm thấy booking")


@tool
def cancel_booking(booking_id: str, reason: str = "") -> str:
    """Huỷ một booking (không thể hoàn tác)."""
    b = db.BOOKINGS.get(booking_id)
    if not b:
        return _err("Không tìm thấy booking")
    if b["status"] == "CANCELLED":
        return _ok(booking_id=booking_id, status="CANCELLED")
    db.FLIGHTS[b["flight_id"]]["seats_left"][b["cabin"]] += len(b["passengers"])
    b["status"] = "CANCELLED"
    return _ok(booking_id=booking_id, status="CANCELLED", reason=reason)


@tool
def request_handoff(reason: str, details: str = "") -> str:
    """Bàn giao cho người (khách hoặc nhân viên) khi KHÔNG thể hoàn thành đúng ràng buộc,
    ví dụ: không có chuyến nào trong ngân sách, thiếu thông tin, lỗi lặp lại. Gọi tool này rồi dừng."""
    return _ok(handoff_requested=True, reason=reason, details=details)


ALL_TOOLS = [search_flights, get_flight_details, hold_booking, process_payment,
             issue_ticket, get_booking, cancel_booking, request_handoff]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
