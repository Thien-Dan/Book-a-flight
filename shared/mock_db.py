"""Cơ sở dữ liệu giả lập (in-memory) cho hệ thống đặt vé máy bay.

Chuyến bay được sinh tất định theo (điểm đi, điểm đến, ngày) nên cùng một truy vấn
luôn ra cùng kết quả -> dễ demo, dễ kiểm thử.
"""
from __future__ import annotations

import hashlib
import random
import string
import threading
from datetime import datetime, timedelta

AIRPORTS = {
    "HAN": "Hà Nội (Nội Bài)",
    "SGN": "TP.HCM (Tân Sơn Nhất)",
    "DAD": "Đà Nẵng",
    "PQC": "Phú Quốc",
    "CXR": "Nha Trang (Cam Ranh)",
    "HPH": "Hải Phòng (Cát Bi)",
    "HUI": "Huế (Phú Bài)",
    "DLI": "Đà Lạt (Liên Khương)",
    "VCA": "Cần Thơ",
    "BKK": "Bangkok (Suvarnabhumi)",
    "SIN": "Singapore (Changi)",
    "ICN": "Seoul (Incheon)",
    "NRT": "Tokyo (Narita)",
}
INTERNATIONAL = {"BKK", "SIN", "ICN", "NRT"}

AIRLINES = {
    "VN": {"name": "Vietnam Airlines", "baggage": "23kg ký gửi + 10kg xách tay", "refund": "Hoàn vé, phí 500.000đ", "factor": 1.35},
    "VJ": {"name": "Vietjet Air", "baggage": "7kg xách tay (ký gửi mua thêm)", "refund": "Không hoàn vé", "factor": 0.85},
    "QH": {"name": "Bamboo Airways", "baggage": "20kg ký gửi + 7kg xách tay", "refund": "Hoàn vé, phí 700.000đ", "factor": 1.1},
    "VU": {"name": "Vietravel Airlines", "baggage": "15kg ký gửi + 7kg xách tay", "refund": "Không hoàn, được đổi ngày", "factor": 0.95},
}

_lock = threading.Lock()
FLIGHTS: dict[str, dict] = {}   # cache các chuyến đã được tìm thấy
BOOKINGS: dict[str, dict] = {}


def _seed(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16)


def generate_flights(origin: str, destination: str, date: str) -> list[dict]:
    rng = random.Random(_seed(origin, destination, date))
    intl = origin in INTERNATIONAL or destination in INTERNATIONAL
    base_minutes = rng.randint(180, 330) if intl else rng.randint(60, 135)
    base_price = rng.randint(2_800_000, 5_500_000) if intl else rng.randint(900_000, 2_000_000)

    flights = []
    for i in range(rng.randint(6, 8)):
        code = rng.choice(list(AIRLINES))
        stops = 1 if rng.random() < 0.25 else 0
        if stops == 0 and code == "VU" and intl:
            code = "VN"  # VU không bay quốc tế thẳng
        dep_h, dep_m = rng.randint(5, 21), rng.choice([0, 15, 30, 45])
        duration = base_minutes + (rng.randint(80, 160) if stops else 0)
        dep = datetime.strptime(f"{date} {dep_h:02d}:{dep_m:02d}", "%Y-%m-%d %H:%M")
        arr = dep + timedelta(minutes=duration)
        eco = int(base_price * AIRLINES[code]["factor"] * rng.uniform(0.8, 1.3) * (0.85 if stops else 1) / 1000) * 1000
        number = f"{code}{rng.randint(100, 999)}"
        fid = f"{number}-{date.replace('-', '')}"
        flights.append({
            "flight_id": fid,
            "flight_no": number,
            "airline": code,
            "airline_name": AIRLINES[code]["name"],
            "origin": origin,
            "destination": destination,
            "date": date,
            "depart_time": dep.strftime("%H:%M"),
            "arrive_time": arr.strftime("%H:%M") + (" (+1)" if arr.date() > dep.date() else ""),
            "duration_min": duration,
            "stops": stops,
            "via": rng.choice(["DAD", "SGN", "HAN"]) if stops else None,
            "prices_vnd": {"economy": eco, "business": int(eco * 2.6 / 1000) * 1000},
            "seats_left": {"economy": rng.choice([0, 1, 2, 3, 4, 6, 9, 9]), "business": rng.choice([0, 0, 1, 2, 4])},
        })
    flights.sort(key=lambda f: f["depart_time"])
    with _lock:
        for f in flights:
            FLIGHTS[f["flight_id"]] = f
    return flights


def _new_id(prefix: str, n: int = 6) -> str:
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def create_hold(flight: dict, cabin: str, passengers: list[str]) -> dict:
    with _lock:
        # Idempotent: cùng chuyến + hạng + danh sách khách -> trả lại booking đang giữ
        for b in BOOKINGS.values():
            if (b["flight_id"] == flight["flight_id"] and b["cabin"] == cabin
                    and sorted(b["passengers"]) == sorted(passengers) and b["status"] in ("HELD", "PAID", "TICKETED")):
                return b
        unit = flight["prices_vnd"][cabin]
        flight["seats_left"][cabin] -= len(passengers)
        booking = {
            "booking_id": _new_id("BK"),
            "flight_id": flight["flight_id"],
            "flight_no": flight["flight_no"],
            "airline": flight["airline"],
            "origin": flight["origin"],
            "destination": flight["destination"],
            "date": flight["date"],
            "depart_time": flight["depart_time"],
            "stops": flight["stops"],
            "cabin": cabin,
            "passengers": list(passengers),
            "unit_price_vnd": unit,
            "total_vnd": unit * len(passengers),
            "status": "HELD",
            "hold_expires_at": (datetime.now() + timedelta(minutes=20)).strftime("%H:%M %d/%m"),
            "payment": None,
            "tickets": [],
        }
        BOOKINGS[booking["booking_id"]] = booking
        return booking
